# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

import tree
import numpy as np


def _tree_batching(trees):
    # batching a list of tree
    # this would convert the structural encoding or decoding problem into a RNN model

    tree_infors = [tree.linear_tree_to_encdec(sample) for sample in trees]
    enc_infor, dec_infor = list(zip(*tree_infors))

    enc_inputs, enc_childs = list(zip(*[list(zip(*e))for e in enc_infor]))
    dec_outputs, dec_ranks, dec_parents = list(zip(*[list(zip(*e))for e in dec_infor]))

    bsz = len(trees)
    max_enc_len = max([len(x) for x in enc_inputs]) + 1
    max_enc_child = max([len(x) for e in enc_childs for x in e])
    max_dec_len = max([len(x) for x in dec_outputs])

    np_enc_inputs = np.zeros([bsz, max_enc_len], dtype=np.int32)
    np_enc_childs = np.zeros([bsz, max_enc_len, max_enc_child], dtype=np.int32)
    np_enc_mask = np.zeros([bsz, max_enc_len], dtype=np.float32)
    np_enc_child_mask = np.zeros([bsz, max_enc_len, max_enc_child], dtype=np.float32)

    np_dec_outputs = np.zeros([bsz, max_dec_len], dtype=np.int32)
    np_dec_ranks  = np.zeros([bsz, max_dec_len], dtype=np.int32)
    np_dec_parents = np.zeros([bsz, max_dec_len], dtype=np.int32)
    np_dec_mask = np.zeros([bsz, max_dec_len], dtype=np.float32)

    for t_idx in range(len(trees)):
        np_enc_inputs[t_idx, 1:len(enc_inputs[t_idx])+1] = list(enc_inputs)[t_idx]
        np_enc_mask[t_idx, 1:len(enc_inputs[t_idx])+1] = 1.0

        np_dec_outputs[t_idx, :len(dec_outputs[t_idx])] = list(dec_outputs)[t_idx]
        np_dec_ranks[t_idx, :len(dec_ranks[t_idx])] = list(dec_ranks)[t_idx]
        np_dec_parents[t_idx, :len(dec_parents[t_idx])] = list(dec_parents)[t_idx]
        np_dec_mask[t_idx, :len(dec_outputs[t_idx])] = 1.0

        for c_idx in range(len(enc_childs[t_idx])):
            np_enc_childs[t_idx, c_idx+1, :len(enc_childs[t_idx][c_idx])] = list(enc_childs)[t_idx][c_idx]
            np_enc_child_mask[t_idx, c_idx+1, :len(enc_childs[t_idx][c_idx])] = 1.0

    return {
               "x": torch.tensor(np_enc_inputs, dtype=torch.long, device='cpu'),              # encoder input, the first-input is meaningless (dummy node)
               "x_c": torch.tensor(np_enc_childs, dtype=torch.long, device='cpu'),            # encoder children, the first-input is meaningless
               "x_m": torch.tensor(np_enc_mask, dtype=torch.float32, device='cpu'),           # mask for sequential dimension
               "x_m_c": torch.tensor(np_enc_child_mask, dtype=torch.float32, device='cpu'),   # mask for children dimension
               "y": torch.tensor(np_dec_outputs, dtype=torch.long, device='cpu'),             # decoder output, the first is also meaningless (root node)
               "y_p": torch.tensor(np_dec_parents, dtype=torch.long, device='cpu'),           # decoder parent, the first one has no parents
               "y_r": torch.tensor(np_dec_ranks, dtype=torch.long, device='cpu'),             # ranking of the current node in its siblings
               "y_m": torch.tensor(np_dec_mask, dtype=torch.float32, device='cpu'),           # mask for sequential dimension
    }


def _batch_indexer(datasize, batch_size):
    # Just divide the datasize into batched size
    dataindex = np.arange(datasize).tolist()

    batchindex = []
    for i in range(datasize // batch_size):
        batchindex.append(dataindex[i * batch_size: (i + 1) * batch_size])
    if datasize % batch_size > 0:
        batchindex.append(dataindex[-(datasize % batch_size):])

    return batchindex


class TreeDataset(object):
    def __init__(self, tree_path, vocab, src_path=None, src_vocab=None,
                 tree_depth_limit=1e8, tree_size_limit=1e8, src_len_limit=1e8):
        super(TreeDataset, self).__init__()
        self.vocab = vocab
        self.tree_depth_limit = tree_depth_limit
        self.tree_size_limit = tree_size_limit
        self.src_len_limit = src_len_limit

        self.use_src = (src_vocab is not None and src_path is not None)
        self.src_vocab = src_vocab

        self.tree_path = os.path.join(tree_path, "tree")
        if self.use_src:
            self.src_path = os.path.join(src_path, "src")

        self.leak_buffer = []

    def load_data(self):
        with open(self.tree_path, 'r') as tree_reader:
            src_reader = open(self.src_path, 'r') if self.use_src else None

            while True:
                tree_line = tree_reader.readline()
                src_line = src_reader.readline() if self.use_src else "<empty>"

                if src_line == "" or tree_line == "":
                    break

                src_line = src_line.strip()
                tree_line = tree_line.strip()

                if src_line == "" or tree_line == "":
                    continue

                # prepare tree structure information
                # remove tree lexical nodes/leaf nodes
                # limit/constrain the tree with some depth/size value
                tree_sample = tree.string_to_tree(tree_line, keep_leaf=False,
                                                  tree_depth_limit=self.tree_depth_limit,
                                                  tree_size_limit=self.tree_size_limit)

                # convert tree labels into ids
                def _label_to_id(x):
                    x.label_id = self.vocab.get_id(x.label)
                tree.get_labels_on_tree(tree_sample, func=_label_to_id)

                if not self.use_src:
                    yield (None, tree_sample)
                else:
                    # prepare source side input feature
                    src_ids = self.src_vocab.to_id(src_line.strip().split()[:self.src_len_limit], append_eos=False)

                    yield (src_ids, tree_sample)

            if self.use_src:
                src_reader.close()

    def to_matrix(self, batch):
        batch_size = len(batch)

        trees = [sample[2] for sample in batch]
        batch_dict = _tree_batching(trees)

        x = [sample[0] for sample in batch]
        batch_dict['id'] = torch.tensor(x, dtype=torch.long, device='cpu')

        if self.use_src:
            src_lens = [len(sample[1]) for sample in batch]
            src_len = max(src_lens)

            s = np.zeros([batch_size, src_len], dtype=np.int32)
            for eidx, sample in enumerate(batch):
                src_ids = sample[1]

                s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]

            batch_dict['s'] = torch.tensor(s, dtype=torch.long, device='cpu')

        return batch_dict

    def batcher(self, size, buffer_size=1000, shuffle=True, train=True):
        def _handle_buffer(_buffer):
            sorted_buffer = sorted(
                _buffer, key=lambda xx: xx[2].size())

            buffer_index = _batch_indexer(len(sorted_buffer), size)
            index_over_index = _batch_indexer(len(buffer_index), 1)

            if shuffle:
                np.random.shuffle(index_over_index)

            for ioi in index_over_index:
                index = buffer_index[ioi[0]]
                batch = [sorted_buffer[ii] for ii in index]
                batch_dict = self.to_matrix(batch)
                batch_dict['raw'] = batch

                yield batch_dict

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, (src_ids, tgt_ids) in enumerate(self.load_data()):
            buffer.append((i, src_ids, tgt_ids))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    batch_size = len(data['raw'])
                    if batch_size < size:
                        self.leak_buffer += data['raw']
                    else:
                        yield data
                buffer = self.leak_buffer
                self.leak_buffer = []

        # deal with data in the buffer
        if len(buffer) > 0:
            for data in _handle_buffer(buffer):
                # check whether the data is tailed
                batch_size = len(data['raw'])
                if train and batch_size < size:
                    self.leak_buffer += data['raw']
                else:
                    yield data
