# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import zglobal
import numpy as np


def saturating_sigmoid(x):
    """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
    y = torch.sigmoid(x)
    return torch.clamp(1.2 * y - 0.1, min=0, max=1)


def inverse_exp_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
    inv_base = np.exp(np.log(min_value) / float(max_step))
    if step is None:
        step = zglobal.global_get('global_step')
    if step is None:
        return 1.0
    step = float(step)
    return inv_base ** np.maximum(float(max_step) - step, 0.0)


def bit_to_int(x_bit, num_bits, base=2):
    """Turn x_bit representing numbers bitwise (lower-endian) to int tensor.
      Args:
        x_bit: Tensor containing numbers in a particular base to be converted to
          int.
        num_bits: Number of bits in the representation.
        base: Base of the representation.
      Returns:
        Integer representation of this number.
    """
    x_l = x_bit.view(-1, num_bits).int().detach()
    x_labels = [
        x_l[:, i] * int(base) ** int(i) for i in range(num_bits)]
    res = sum(x_labels)
    return res.view(x_bit.size()[:-1]).int()


class ISemHash(nn.Module):
    def __init__(self, in_dim, z_dim, filter_dim,
                 noise_dev=0.5, startup_steps=50000, discrete_mix=0.5):
        """
            discrete_mix: Factor for mixing discrete and non-discrete input. Used only
                if bottleneck_kind is semhash.
            noise_dev: Noise stddev. Used only if bottleneck_kind is semhash.
            startup_steps: Number of steps after which latent predictor is trained. Used
                only if bottleneck_kind is semhash.
        """
        super(ISemHash, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.filter_dim = filter_dim

        self.noise_dev = noise_dev
        self.startup_steps = startup_steps
        self.discrete_mix = discrete_mix

        self.i_to_z = nn.Linear(in_dim, z_dim)

        self.o_dense_a = nn.Linear(z_dim, filter_dim)
        self.o_dense_b = nn.Linear(z_dim, filter_dim)
        self.o_dense   = nn.Linear(filter_dim, in_dim)

    def forward(self, inputs):
        """https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/discretization.py#L480"""

        outputs_discrete = self.i_to_z(inputs)
        y_clean = saturating_sigmoid(outputs_discrete)
        if self.noise_dev > 0 and self.training:
            # problem: this is a normal distribution but not truncated_normal
            noise = outputs_discrete.new_zeros(outputs_discrete.size()).normal_(0.0, self.noise_dev)
            y = saturating_sigmoid(outputs_discrete + noise)
        else:
            y = y_clean

        d = (0.5 < y).float()
        y_discrete = d.detach() + y - y.detach()

        pd = inverse_exp_decay(self.startup_steps * 2)
        pd *= self.discrete_mix
        pd = pd if self.training else 1.0

        uniform_noise = y.new_zeros([y.size(0), 1]).uniform_()
        uniform_mask = (uniform_noise < pd).float()
        c = uniform_mask * y_discrete + (1. - uniform_mask) * y

        outputs_dense_a = self.o_dense_a(c)
        outputs_dense_b = self.o_dense_b(1.0 - c)
        outputs_dense = self.o_dense(F.relu(outputs_dense_a + outputs_dense_b))
        outputs_discrete = bit_to_int(d, self.z_dim)

        return outputs_dense, outputs_discrete
