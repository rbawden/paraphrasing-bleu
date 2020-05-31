# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


# Multi-Head Attention Module
class MHAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MHAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.d_qkv = d_model // n_head

        # query, key, value layer mapping
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.mapper = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        d_qkv, n_head = self.d_qkv, self.n_head

        # B x H x T_ x D_
        q = self.query(q).view(sz_b, len_q, n_head, d_qkv)
        k = self.key(k).view(sz_b, len_k, n_head, d_qkv)
        v = self.value(v).view(sz_b, len_v, n_head, d_qkv)

        # scaled down query for stabilization
        q *= d_qkv ** -0.5

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_qkv)   # (H*B) x Tq x Dq
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_qkv)   # (H*B) x Tk x Dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_qkv)   # (H*B) x Tv x Dv

        # scaled dot for attention map
        attn_logits = torch.bmm(q, k.transpose(1, 2))

        # to cope with paddings
        if mask is not None:
            attn_logits += mask

        attn_weights = self.softmax(attn_logits)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_weights, v)

        output = attn_output.view(n_head, sz_b, len_q, d_qkv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # B x Tq x D

        output = self.mapper(output)

        return output, attn_weights


# Feed-Forward Layer with 2 internal layers
class FFNLayer(nn.Module):
    def __init__(self, d_model, d_hid, dropout=0.1):
        super(FFNLayer, self).__init__()

        self.i2h = nn.Linear(d_model, d_hid)
        self.h2o = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.relu(self.i2h(x))
        hidden = self.dropout(hidden)
        output = self.h2o(hidden)

        return output


# Generate different attentional mask
def attn_mask_maker(mask, mode="masking", inf=-1e8):
    if mode == "masking":
        # masking: full self-attention or encoder-decoder attention
        ret = (1. - mask) * inf
        # B x 1 x T
        return ret.unsqueeze(1)
    else:
        raise ValueError("Unknown mode %s" % mode)


# Apply Sinusoid positional embedding to input x
def sinusoid_position_encoder(x, min_timescale=1.0, max_timescale=1.0e4):
    sz_b, len_x, d_x = x.size()

    position = torch.arange(len_x, dtype=x.dtype, device=x.device)
    num_timescales = d_x // 2

    log_timescale_increment = (
            np.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=x.dtype, device=x.device) * -log_timescale_increment
    )

    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    signal = signal.view(1, len_x, d_x).contiguous()

    return x + signal


# Add Residual Connection
class ResidualLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # shift it to RMSNorm?
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, y):
        y = self.dropout(y)
        o = x + y
        return self.layer_norm(o)


# Transformer Self-atention based Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_model, d_inner,
                 relu_dropout=0.1, res_dropout=0.1, attn_dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_inner = d_inner

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_dropout = attn_dropout

        # self-attention module
        self.san_layer_stack = nn.ModuleList([
            MHAttention(n_head, d_model, dropout=attn_dropout) for _ in range(n_layers)
        ])

        # feed-forward module
        self.ffn_layer_stack = nn.ModuleList([
            FFNLayer(d_model, d_inner, dropout=relu_dropout) for _ in range(n_layers)
        ])

        # residual & layer normalization module
        self.res_layer_stack = nn.ModuleList([
            ResidualLayer(d_model, dropout=res_dropout) for _ in range(n_layers * 2)
        ])

    def forward(self, x, mask=None):
        # formulate mask
        if mask is not None:
            mask = attn_mask_maker(mask, mode="masking")
            mask = mask.repeat(self.n_head, 1, 1)

        # add positional encoding
        x = sinusoid_position_encoder(x)

        # layer stack
        for layer in range(self.n_layers):
            # self-attention layer
            y, w = self.san_layer_stack[layer](x, x, x, mask=mask)
            # residual connection
            x = self.res_layer_stack[2 * layer](x, y)

            # ffn layer
            y = self.ffn_layer_stack[layer](x)
            # residual connection
            x = self.res_layer_stack[2 * layer + 1](x, y)

        return x
