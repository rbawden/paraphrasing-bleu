#!/usr/bin/env python3

"""
Takes system outputs and references and computes

if more than one system:
- the set of unique n-grams (found in the reference) in each system

if more than one reference:
- the set of n-grams matched in references 2+ not found in ref1
"""

import os
import sacrebleu
import sys

from collections import Counter

def main(args):

    if args.human_scores:
        systems = []
        scores = {}
        for line in open(args.human_scores):
            system, score = line.rstrip().split()
            scores[system] = float(score)
        for system in args.systems:
            system_name = '.'.join(os.path.basename(system).split('.')[1:-1])
            if system_name not in scores:
                print(f"COULDN'T FIND SYSTEM {system_name}", file=sys.stderr)
            elif scores[system_name] >= args.human_min and scores[system_name] <= args.human_max:
                systems.append(system)
    else:
        systems = args.systems

    sys_fhs = [open(ref) for ref in systems]
    ref_fhs = [open(system) for system in args.refs]

    tokenize = sacrebleu.TOKENIZERS["13a"]

    stats = Counter()
    totals = Counter()
    for lineno, (syss, refs) in enumerate(zip(zip(*sys_fhs), zip(*ref_fhs)), 1):
        syss = [tokenize(system) for system in syss]
        refs = [tokenize(ref) for ref in refs]

        # All system n-grams.
        sys_ngrams = Counter()
        for system in syss:
            sys_ngrams += sacrebleu.extract_ngrams(system, max_order=args.n)

        for ngram in sys_ngrams.keys():
            totals[len(ngram.split())] += 1

        # reset counts for all n-grams found in references
        ref_ngrams = Counter()
        for ref in refs:
            ref_ngrams += sacrebleu.extract_ngrams(ref, max_order=args.n)

        ngrams = list(sys_ngrams.keys())
        for ngram in ngrams:
            if ngram not in ref_ngrams:
                del sys_ngrams[ngram]

        for ngram in sys_ngrams.keys():
            stats[len(ngram.split())] += 1

    def pretty(counter: Counter):
        return '\t'.join(f'{counter[x]}' for x in range(1, args.n + 1))

    stats = [stats[x] / totals[x] for x in range(1, args.n + 1)]
    print('UNSEEN:', *[f"{x*100:.2f}" for x in stats], sep='\t')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', '-s', nargs='+')
    parser.add_argument('--refs', '-r', nargs='+')
    parser.add_argument('--human-scores', help='file with human scores')
    parser.add_argument('--human-min', type=int, default=0, help="human score minimum")
    parser.add_argument('--human-max', type=int, default=100, help="human score maximum")
    parser.add_argument('-n', type=int, default=4)
    args = parser.parse_args()

    main(args)
