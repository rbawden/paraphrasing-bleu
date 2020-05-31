# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import attn
import util
import bottleneck


# child sum LSTM cell, for bottom-up composition
class ChildSumLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumLSTMCell, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def forward(self, x, x_c, x_m_c):
        # child sum memory cell for processing one time step
        # x: node label representation, [batch, dim]
        # x_c: children representation [batch, children_number, dim]
        # x_m_c: children mask, [batch, children_number, dim]
        #   this mask can also used for averaging children node representation
        # state order: memory cell + hidden state

        x_h = x
        x_c_s, x_c_h = torch.split(x_c, self.mem_dim, dim=-1)

        # [batch, dim]
        child_h_sum = torch.sum(x_c_h * x_m_c, dim=1)

        iou = self.ioux(x_h) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        # [batch, children_number, dim]
        f = torch.sigmoid(
            self.fh(x_c_h) +
            util.expand_tile_dims(self.fx(x_h), x_c.size(1), axis=1)
        )
        fc = torch.mul(f, x_c_s)

        c = torch.mul(i, u) + torch.sum(fc * x_m_c, dim=1)
        h = torch.mul(o, torch.tanh(c))

        # [batch, dim]
        return torch.cat([c, h], dim=-1)


# enable linear mapping for branch/partition-based calculation
class BranchLinear(nn.Module):
    def __init__(self, in_features, out_features, branch=1, bias=True):
        super(BranchLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.branch = branch

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, branch))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, branch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # this maybe problematic, because initialization is changed
        fan_in = self.in_features
        nonlinearity = 'leaky_relu'

        gain = nn.init.calculate_gain(nonlinearity, math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, order):
        # input: the input tensor, of shape [batch, *, dim_in]
        # order: the order feature, of shape [batch]

        # extract order-relevant weights and bias
        order = order % self.branch
        # [batch, branch]
        one_hot_order = torch.eye(self.branch, device=order.device).float()[order]

        # extract weight and bias for order-specific linear mapping
        # [out_feature, in_feature, batch]
        sel_weight = self.weight.matmul(one_hot_order.transpose(0, 1))
        # [batch, in_featuere, out_feature]
        sel_weight = sel_weight.permute(2, 1, 0)

        in_shp = input.size()
        # [batch, ?, dim_in]
        input = input.view(in_shp[0], -1, in_shp[-1])

        # [batch, ?, out_feature]
        output = input.matmul(sel_weight)

        # consider the bias term
        if self.bias is not None:
            # [out_feature, batch]
            sel_bias = self.bias.matmul(one_hot_order.transpose(0, 1))
            # [batch, 1, out_feature]
            sel_bias = sel_bias.permute(1, 0).unsqueeze(1)

            output = output + sel_bias

        # reshape back the input shape & dimension
        out_shp = list(in_shp)
        out_shp[-1] = self.out_features
        # [batch, *, dim_out]
        output = output.view(*out_shp)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, branch={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.branch
        )


# parent expand LSTM cell, for top-down decomposition
class ParentExpandLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim, max_branch):
        super(ParentExpandLSTMCell, self).__init__()
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.max_branch = max_branch

        self.iouh = BranchLinear(self.mem_dim, 3 * self.mem_dim, branch=max_branch)
        self.fh = BranchLinear(self.mem_dim, self.mem_dim, branch=max_branch)

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)

    def forward(self, y, y_r, y_p):
        # y: the parent label embedding, shape: [batch, dim]
        # y_r: ranking information, rank in children, shape [batch]
        # y_p: the parent node representation, shape [batch, dim]
        # state order: memory cell + hidden state

        y_h = y
        y_p_s, y_p_h = torch.split(y_p, self.mem_dim, dim=-1)

        iou = self.ioux(y_h) + self.iouh(y_p_h, y_r)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        # [batch, dim]
        f = torch.sigmoid(
            self.fh(y_p_h, y_r) + self.fx(y_h)
        )
        fc = torch.mul(f, y_p_s)

        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))

        # [batch, dim]
        return torch.cat([c, h], dim=-1)


# tree-based LSTM encoder
class TreeLSTMEncoder(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(TreeLSTMEncoder, self).__init__()
        self.mem_dim = mem_dim

        self.cscell = ChildSumLSTMCell(in_dim, mem_dim)

    # postorder traversal
    def forward(self, x, x_c, x_m, x_m_c, init_state=None):
        # x: tensor sequence with shape [batch, length, dim]
        # x_c: tensor children sequence with [batch, length, child_number]
        # x_m: masked inputs with shape [batch, length]
        # x_m_c: masked children inputs ith shape [batch, length, child_number]
        # init_state: [batch, dim * 2]: LSTM state, one for hidden state and one for memory cell
        # state order: memory cell + hidden state

        bsz, len_seq, num_child = x_c.size()[:3]

        if init_state is None:
            init_state = x.new_zeros((bsz, self.mem_dim * 2))

        if x_m is None:
            x_m = x.new_ones((bsz, len_seq))
        if x_m_c is None:
            x_m_c = x.new_ones((bsz, len_seq, num_child))

        outputs = x.new_zeros((bsz, len_seq, self.mem_dim * 2))
        outputs[:, 0] = init_state

        prev_state = init_state

        for step in range(1, len_seq):
            index = step

            x_t = x[:, index]       # [batch, dim]
            x_c_t = x_c[:, index]   # [batch, children_number]
            x_m_t = x_m[:, index].unsqueeze(-1)  # [batch, 1]
            x_m_c_t = x_m_c[:, index].unsqueeze(-1)  # [batch, child_number, 1]

            # extract children node representation
            # shape: [batch, children_number, dim]
            x_c_repr = outputs.clone().gather(
                1,
                util.expand_tile_dims(
                    x_c_t, outputs.size(-1), axis=2
                )
            )
            # shape: [batch, dim]
            s = self.cscell(x_t, x_c_repr, x_m_c_t)

            # mixing up with masking, copy root state at the end
            s = x_m_t * s + (1. - x_m_t) * prev_state

            prev_state = prev_state.clone()
            prev_state[:] = s
            outputs[:, index] = s

        # [batch, dim]
        final_state = prev_state

        return outputs, final_state


# tree-based LSTM decoder
class TreeLSTMDecoder(nn.Module):
    def __init__(self, in_dim, mem_dim, max_branch=3):
        super(TreeLSTMDecoder, self).__init__()
        self.max_branch = max_branch
        self.mem_dim = mem_dim
        self.in_dim = in_dim

        self.pecell = ParentExpandLSTMCell(in_dim, mem_dim, max_branch)

    # preorder traversal
    def forward(self, y, y_p, y_r, y_m, init_state=None):
        # y: tensor sequence, current id list, shape: [batch, length, dim]
        # y_p: tensor sequence, parent pointer, shape: [batch, length]
        # y_r: tensor sequence, rank of sibling, shape: [batch, length]
        # y_m: mask matrix, shape: [batch, length]
        # init_state: [batch, dim * 2] : LSTM state, one for hidden state and one for memory cell
        # state order: memory cell + hidden state

        bsz, len_seq = y_p.size()

        assert init_state is not None, 'Autoencoder requires decoding initial states'

        if y_m is None:
            y_m = init_state.new_ones((bsz, len_seq))

        outputs = init_state.new_zeros((bsz, len_seq, self.mem_dim * 2))
        outputs[:, 0] = init_state

        for step in range(1, len_seq):
            index = step

            y_p_t = y_p[:, index].unsqueeze(-1)   # [batch, 1]
            y_r_t = y_r[:, index]   # [batch]
            y_m_t = y_m[:, index].unsqueeze(-1)   # [batch, 1]

            # extract parent node representation
            # shape: [batch, 1, dim]
            # lesson: don't use tensor.data.clone(). `data` causes no gradient!
            y_p_repr = outputs.clone().gather(
                1,
                util.expand_tile_dims(
                    y_p_t, outputs.size(-1), axis=2
                )
            )
            y_p_repr = y_p_repr.squeeze(1)

            # extract parent node label embedding
            y_embed = y.clone().gather(
                1,
                util.expand_tile_dims(
                    y_p_t, y.size(-1), axis=2
                )
            )
            y_embed = y_embed.squeeze(1)

            # shape: [batch, dim]
            s = self.pecell(y_embed, y_r_t, y_p_repr)

            # mixing up with masking, copy parent node representation
            s = y_m_t * s + (1. - y_m_t) * y_p_repr

            outputs[:, index] = s

        return outputs


# the auto-encoder model
class TreeLSTMAutoEncoder(nn.Module):
    def __init__(self,
                 # tree model parameter
                 vocab_size,
                 in_dim,
                 mem_dim,
                 bits_number=8,
                 filter_size=2048,
                 max_num_children=3,
                 # improved semantic hashing parameter
                 noise_dev=0.5,
                 startup_steps=50000,
                 discrete_mix=0.5,
                 use_bottleneck=True,
                 # source encoding parameter
                 src_vocab_size=None,
                 atn_num_layer=4,
                 atn_num_heads=4,
                 atn_dp=0.1,
                 ):
        super(TreeLSTMAutoEncoder, self).__init__()
        self.mem_dim = mem_dim
        self.use_bottleneck = use_bottleneck
        self.use_src = (src_vocab_size is not None)

        # node label & source word embedding layer
        self.emb = nn.Embedding(vocab_size, in_dim)
        if self.use_src:
            self.src_emb = nn.Embedding(src_vocab_size, mem_dim)

        # tree-based encoder & decoder
        self.tree_encoder = TreeLSTMEncoder(in_dim, mem_dim)
        self.tree_decoder = TreeLSTMDecoder(in_dim, mem_dim, max_branch=max_num_children)

        # internal bottleneck layer
        # improved semantic hashing or simple linear mapping
        #   the latter one requires clustering algorithm
        if use_bottleneck:
            self.discretization = bottleneck.ISemHash(
                mem_dim, bits_number, filter_size,
                noise_dev=noise_dev, startup_steps=startup_steps, discrete_mix=discrete_mix)
        else:
            self.discretization = nn.Linear(mem_dim, mem_dim)

        # initialization layer for decoder
        self.decoder_init = nn.Linear(mem_dim, 2 * mem_dim)

        # mapping hidden representations for target-structure prediction
        self.output_layer = nn.Linear(mem_dim, vocab_size, bias=False)

        # embedding and softmax weight sharing
        if in_dim == mem_dim:
            self.output_layer.weight = self.emb.weight

        # source-side transformer encoder
        if self.use_src:
            self.src_encoder = attn.TransformerEncoder(
                atn_num_layer, atn_num_heads, mem_dim, mem_dim * 4,
                res_dropout=atn_dp, relu_dropout=atn_dp, attn_dropout=0.0
            )

    def forward(self, inputs):
        # inputs: a dictionary, extract useful information from it
        if self.use_src:
            assert 's' in inputs, "source sentence must be given"

            src_inputs = inputs['s']
            src_mask = (src_inputs > 0).float()

            src_embed = self.src_emb(src_inputs)
            src_encoding = self.src_encoder(src_embed, mask=src_mask)

            src_encoding = src_encoding.sum(1) / src_mask.sum(1, keepdim=True)

        x = self.emb(inputs['x'])
        y = self.emb(inputs['y'])
        x_c = inputs['x_c']
        x_m = inputs['x_m']
        x_m_c = inputs['x_m_c']
        y_p = inputs['y_p']
        y_r = inputs['y_r']
        y_m = inputs['y_m']

        # encoding
        _, tree_encoding = self.tree_encoder(x, x_c, x_m, x_m_c)
        _, tree_encoding = torch.split(tree_encoding, self.mem_dim, -1)

        if self.use_src:
            tree_encoding += src_encoding

        # bottleneck
        if self.use_bottleneck:
            tree_repr, tree_codes = self.discretization(tree_encoding)
        else:
            tree_repr = self.discretization(tree_encoding)
            tree_codes = tree_repr.new_ones(tree_repr.size(0))
        otree_repr = tree_repr

        if self.use_src:
            tree_repr += src_encoding

        # decoding
        root_dec_state = self.decoder_init(tree_repr)
        tree_state = self.tree_decoder(y, y_p, y_r, y_m, init_state=root_dec_state)

        # extract ordered decoding state for prediction
        _, tree_state = torch.split(tree_state, self.mem_dim, -1)

        # predicting the linear_inputs
        logits = self.output_layer(tree_state)

        return logits, tree_codes, otree_repr
