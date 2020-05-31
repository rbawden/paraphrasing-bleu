# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from nltk import Tree as NLTKTree


class Tree(object):
    def __init__(self):
        self.parent = None          # parent node, None: root node
        self.num_children = 0       # children nodes number
        self.children = []          # the children nodes
        self.label = None           # label, such as S, VP, NP
        self.label_id = None        # label's id in vocabulary
        self.index = None           # index, convert the nodes into linearized structure, index number
        self.root_depth = -1        # depth from root

        self._size = -1             # sub-tree size, all node number
        self._depth = -1            # depth from bottom
        self._str = None            # string representation
        self.rank_in_children = -1  # the ranking in all children, counting sibling nodes

    def add_child(self, child):
        child.parent = self
        child.rank_in_children = self.num_children
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if self._size < 0:
            count = 1
            count += sum(child.size() for child in self.children)
            self._size = count

        return self._size

    def depth(self):
        if self._depth < 0:
            count = 0
            if self.num_children > 0:
                count = max(child.depth() for child in self.children)
                count += 1
            self._depth = count

        return self._depth

    def __str__(self):
        if self._str is None:
            str_list = []
            if self.num_children > 0:
                str_list.append('({}'.format(self.label))
                for child in self.children:
                    str_list.append(str(child))
                self._str = ' '.join(str_list) + ')'
            else:
                self._str = self.label

        return self._str


def string_to_tree(tree_string, keep_leaf=True, tree_depth_limit=1e8, tree_size_limit=1e8):
    # convert parsing linearized string into tree object
    # do not save the lexical node as stated in the paper

    # pre-process tree string, to handle exceptions
    # Replace leaves of the form (word) with word
    tree_string = re.sub(r"\(([^\s()]+)\)", r"\1", tree_string)

    # keep_leaf: in our case, we remove the lexical string, so set it to False
    # tree_depth_limit: when you want to limit the tree structure, use it
    #   only keep the top-level structure of certain depth
    # tree_size_limit: when you want to limit the tree structure, use it
    #   only keep the top-level structure of certain node number
    nltk_tree = NLTKTree.fromstring(tree_string)

    tree = Tree()
    tree.label = nltk_tree.label()
    tree.root_depth = 0

    def copy_from_nltk_tree(_from, _to, _dep):
        # out of tree depth limit, pass directory
        if _dep < tree_depth_limit:

            for child in _from:
                node = Tree()
                node.root_depth = _dep

                if isinstance(child, NLTKTree):
                    node.label = child.label()
                    if copy_from_nltk_tree.tree_size < tree_size_limit:
                        copy_from_nltk_tree.tree_size += 1
                        _to.add_child(node)
                    copy_from_nltk_tree(child, node, _dep+1)
                else:
                    node.label = child

                    if keep_leaf:
                        if copy_from_nltk_tree.tree_size < tree_size_limit:
                            copy_from_nltk_tree.tree_size += 1
                            _to.add_child(node)

    copy_from_nltk_tree.tree_size = 1
    copy_from_nltk_tree(nltk_tree, tree, 1)

    return tree


def get_labels_on_tree(tree,
                       func=lambda x: x.label,
                       use_leaf=True, use_non_terminal=True,
                       index_tree=False, index_base=0,
                       preorder=True):
    # extract labels or states from tree nodes
    # you can use it to extract tree information, in a preorder or postorder manner

    # func: the information you can extract from one node, you can define it by yourself
    # use_leaf: whether include leaf node
    # use_non_terminal: whether include non-terminal node
    # index_tree: whether index the tree according to traversal order
    # index_base: the start point for indexing
    # preorder: preorder traversal or postorder traversal

    def _get_labels_on_tree(_tree):
        nodes = []

        if preorder:
            if (_tree.num_children > 0 and use_non_terminal) \
                    or (_tree.num_children <= 0 and use_leaf):
                nodes.append(_tree)

        for child in _tree.children:
            nodes += _get_labels_on_tree(child)

        if not preorder:
            if (_tree.num_children > 0 and use_non_terminal) \
                    or (_tree.num_children <= 0 and use_leaf):
                nodes.append(_tree)

        return nodes

    # order is used for indexing the tree nodes
    ordered_nodes = _get_labels_on_tree(tree)

    if index_tree:
        for n_idx in range(len(ordered_nodes)):
            ordered_nodes[n_idx].index = n_idx + index_base

    return [func(node) for node in ordered_nodes]


def linear_tree_to_encdec(tree):
    # given a tree, linearized it for batching, including encoding and decoding

    # for batching a tree, information required is as follows:
    #   1. a ordered node list, preorder for encoding and post order for decoding
    #   2. a parent or children list, get parent or children information for composition or decomposition
    #   3. children index to distinguish composition

    # extract encoding information
    enc_postorder_nodes = get_labels_on_tree(tree, func=lambda x: x, preorder=False, index_tree=True, index_base=1)
    # label: for input string, index: for children information extraction
    enc_infor = [[node.label_id, [x.index for x in node.children]] for node in enc_postorder_nodes]

    # extract decoding information
    dec_preorder_nodes = get_labels_on_tree(tree, func=lambda x: x, preorder=True, index_tree=True, index_base=0)
    # label: for output string, rank_in_children: for decomposition cell, index: for parent information extraction
    dec_infor = [[node.label_id, node.rank_in_children,
                  0 if node.parent is None else node.parent.index] for node in dec_preorder_nodes]

    # for encoder, the index is 1-based, the 0-index is for dummy inputs
    # batching need padding and masking
    return enc_infor, dec_infor
