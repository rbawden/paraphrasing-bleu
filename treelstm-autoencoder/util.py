# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F


def expand_tile_dims(x, depth, axis=1):
    """Expand and Tile x on axis by depth"""
    x = x.unsqueeze(axis)
    tile_dims = [1] * len(x.size())
    tile_dims[axis] = depth

    return x.repeat(tile_dims)


def masked_loss(logits, target, mask=None):
    """MLE loss with mask"""
    # logits: [batch, sequence, vocab_size]
    # target: [batch, sequence]
    # mask: [batch, sequence]

    flat_logits = logits.view(-1, logits.size(-1))
    flat_target = target.view(-1, 1)

    soft_targets = logits.new_zeros(
        flat_logits.size()).scatter_(-1, flat_target, 1.)

    mle_loss = - F.log_softmax(flat_logits, -1) * soft_targets
    mle_loss = mle_loss.sum(-1).view(*target.size())

    if mask is not None:
        mle_loss = (mle_loss * mask).sum(-1) / mask.sum(-1)
    else:
        mle_loss = mle_loss.sum(-1)

    return mle_loss


def masked_acc(logits, target, mask=None):
    """accuracy with mask"""
    # logits: [batch, sequence, vocab_size]
    # target: [batch, sequence]
    # mask: [batch, sequence]

    predicts = logits.argmax(-1)

    corr_pred = (predicts == target).float()

    if mask is not None:
        corr_pred = corr_pred * mask

    corr_count = corr_pred.sum()
    total_count = corr_pred.numel()

    if mask is not None:
        total_count = mask.sum()

    acc = corr_count * 100. / total_count

    return corr_count, total_count, acc
