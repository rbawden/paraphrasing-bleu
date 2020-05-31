#!/usr/bin/python
from syntactic_similarity import *
import sys
import re

def prune_trees(tree_file, depth=-1, remove_leaves=False):
    for line in tree_file:
        # get tree
        tree = Tree.fromstring(line.strip())
        
        # get rid of leaves
        if remove_leaves:
            tree =  delete_leaves(tree)            
        # prune the tree
        if depth > -1:
            tree = delete_all_below_depth(tree, depth)
            
        # print out
        print(re.sub(' +', ' ', str(tree).replace('\n', ' ')), flush=True)
            
        
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parse_tree_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--depth', '-d', type=int, default=-1)
    parser.add_argument('--remove_leaves', '-r', action='store_true', default=False)
    args = parser.parse_args()

    prune_trees(args.parse_tree_file, args.depth, args.remove_leaves)
