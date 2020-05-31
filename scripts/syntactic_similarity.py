#!/usr/bin/python
from nltk.tree import *
import numpy as np
from apted import APTED, helpers
from nltk import Tree
import re
import os

def apted_tree_format(tree):
    '''
    Convert an nltk tree to apted string format
    Args: 
      - tree: nltk tree
    Returns:
      - strtree: string tree format of the form {a{b}{c}}
    '''
    if type(tree)==str:
        return " {" + tree + "} "
    else:
        strtree = " {" + tree.label()

        for child in tree:
            strtree += apted_tree_format(child)
        return strtree + "} "


def apted(tree1, tree2):
    # remove outer brackets and strip all white space
    str_t1 = apted_tree_format(tree1).strip()[1:-1].strip()
    str_t2 = apted_tree_format(tree2).strip()[1:-1].strip()

    # convert to apted tree from apted format
    t1 = helpers.Tree.from_text(str_t1)
    t2 = helpers.Tree.from_text(str_t2)

    apted = APTED(t1, t2)

    return apted.compute_edit_distance()



def rule_intersection(tree1, tree2, norm_length=True):
    '''
    Computes the length of the intersection between the production
    rules in tree1 and tree2 (both nltk trees).
    By default normalises for the total number of productions.
    This can be switched off if norm_lenth = False
    '''
    p1 = set(tree1.productions())
    p2 = set(tree2.productions())

    len_inter = len(p1.intersection(p2))

    if norm_length:
        denom = (len(p1) + len(p2)) / 2
        return len_inter / denom
    else:
        return len_inter


def compare_trees(tree1, tree2, method):
    '''
    Compare 2 NLTK trees using the specified method
    '''

    # Tree Edit distance using APTED
    if method == 'ted':
        return apted(tree1, tree2)

    # (Normalised) length of intersection of CFG productions
    elif method == 'rule_intersection':
        return rule_intersection(tree1, tree2)


    else:
        exit("No comparison method specified")


def read_trees(parse_tree_file):
    '''
    Read a file containing one tree per line of the format 
    (S (NP I) (VP (V enjoyed) (NP my cookie))) and converts
    into NLTK tree
    '''
    trees = []
    with open(parse_tree_file) as fp:
        for line in fp:
            tree = Tree.fromstring(line.strip())
            trees.append(tree)
    return trees


def test_trees():
    '''
    Return a list of example trees to test out functions
    '''
    trees = ['(S (NP I) (VP (V enjoyed) (NP my cookie)))',
             '(s (dp (d the) (np dog)) (vp (v chased) (dp (d the) (np cat))))']
    for i in range(len(trees)):
        trees[i] = Tree.fromstring(trees[i])
        trees[i].pretty_print(unicodelines=True, nodedist=4)
        print(trees[i].label())
    return trees

# input & output is nltk.Tree
# the leaf is always only child (or other leaves) and tag is new terminal node
# nltk.Tree finds leaves with instance(tree, Tree), I used the same logic
def delete_leaves(tree):
    for c, child in enumerate(tree):

        if not isinstance(child, Tree):  # is a leaf & there is no other children
            return tree.label()
        else:
            tree[c] = delete_leaves(child)
    return tree


# input & output is nltk.Tree
# depth below one means only root
def delete_all_below_depth(tree, depth_below):
    # if we have leaf, no need to go further
    if not isinstance(tree, Tree):
        return tree

    if depth_below == 0:
        return tree.label()  # making the current node terminal, forgetting everything below
    else:
        for c, child in enumerate(tree):
            tree[c] = delete_all_below_depth(child, depth_below - 1)  # moving deeper, too early for a cut
        return tree


def compare_all_trees(trees, method):

    os.sys.stderr.write('Computing similarity using: ' + method + '.\n')
    if method in ["ted"]:
        os.sys.stderr.write('This is a distance metric so the smaller the valuethe better!\n')
    else:
        os.sys.stderr.write('This is a similarity metric so the greater the value the better!\n')

    similarities = np.zeros((len(trees), len(trees))) # similarity matrix
    for t1, tree1 in enumerate(trees):
        for t2, tree2 in enumerate(trees):
            similarity = compare_trees(tree1, tree2, method=method)
            similarities[t1, t2] = similarity

    print(similarities)




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parse_tree_file')
    parser.add_argument('method', choices=["ted", "rule_intersection"])
    args = parser.parse_args()

    trees = read_trees(args.parse_tree_file)
    #trees = test_trees() # load up some test trees
    compare_all_trees(trees, args.method)
