# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM-based Autoencoder for Generate Syntactic Sentence codes')

    # data arguments
    parser.add_argument('--data', default='data/',
                        help='path to dataset')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='model',
                        help='Name to identify experiment')

    # source arguments
    parser.add_argument('--use_src', action='store_true',
                        help='whether use source sentence')
    parser.add_argument('--num_layer', default=3, type=int,
                        help='number of transformer layers')
    parser.add_argument('--num_head', default=4, type=int,
                        help='number of attention heads')
    parser.add_argument('--atn_dp', default=0.1, type=float,
                        help='dropout rate inside attention')

    # model arguments
    parser.add_argument('--input_dim', default=256, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=256, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--max_num_children', default=10, type=int,
                        help='Maximum children allowed for each node')

    # bottleneck arguments / improved semantic hashing
    parser.add_argument('--bit_number', default=8, type=int,
                        help='Number of bits for semantic hashing')
    parser.add_argument('--filter_size', default=2048, type=int,
                        help='Size of hashing internal state')
    parser.add_argument('--startup_size', default=10000, type=int,
                        help='The step number of peak point for hashing')
    parser.add_argument('--noise_dev', default=0.5, type=float,
                        help='Standard deviation for noise injection')
    parser.add_argument('--discrete_mix', default=0.5, type=float,
                        help='Mixture probability for training hashing')
    parser.add_argument('--use_bottleneck', action='store_true',
                        help='Whether use discretization technique')

    # training arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--buffersize', default=1024, type=int,
                        help='number of batches used for shuffling')
    parser.add_argument('--num_grad_agg', default=1, type=int,
                        help='number of gradient aggregation')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--mode', choices=['train', 'eval'],
                        default='train',
                        help='train or evaluate the model')

    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('--disp_freq', default=1, type=int,
                        help='display frequency during training')
    parser.add_argument('--save_freq', default=5000, type=int,
                        help='save checkpoint frequency during training')
    parser.add_argument('--device', nargs='+', type=int,
                        help='the index of gpu device')
    parser.add_argument('--max_depth', default=1e10, type=int,
                        help='Largest depth to encoding/decoding the tree')
    parser.add_argument('--max_tree_size', default=1e10, type=int,
                        help='Maximum tree size, i.e. node number')
    parser.add_argument('--max_src_len', default=1e10, type=int,
                        help='Maximum source sentence length')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
