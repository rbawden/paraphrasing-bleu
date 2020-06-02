#!/usr/bin/env python3

"""
Takes system outputs and references and computes

if more than one system:
- the set of unique n-grams (found in the reference) in each system

if more than one reference:
- the set of n-grams matched in references 2+ not found in ref1
"""

import sacrebleu
import sys

from collections import Counter

def main(args):

    sys_fhs = [open(ref) for ref in args.systems]
    ref_fhs = [open(sys) for sys in args.refs]

    stats = Counter()
    for lineno, (syss, refs) in enumerate(zip(zip(*sys_fhs), zip(*ref_fhs)), 1):
        syss = [sacrebleu.tokenize_13a(sys) for sys in syss]
        refs = [sacrebleu.tokenize_13a(ref) for ref in refs]

        # Find all ngrams in refs#2+ that are not in ref#1
        ref_ngrams = Counter()
        for ref in refs[1:]:
            ref_ngrams += sacrebleu.extract_ngrams(ref, max_order=args.n)
        for ngram in sacrebleu.extract_ngrams(refs[0], max_order=args.n):
            if ngram in ref_ngrams:
                del ref_ngrams[ngram]

        # The system ngrams that are only in refs 2+
        sys_ngrams = [sacrebleu.extract_ngrams(sys, max_order=args.n) for sys in syss]
        if len(syss) == 1:
            for ngram in sys_ngrams[0]:
                if ngram not in ref_ngrams:
                    sys_ngrams[0][ngram] = 0
        sys_ngrams[0] += Counter()

        for ngram in sys_ngrams[0]:
            stats[len(ngram.split())] += sys_ngrams[0][ngram]

    def pretty(counter: Counter):
        return '\t'.join(f'{counter[x]}' for x in range(1, args.n + 1))


    print(pretty(stats))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', '-s', nargs='+')
    parser.add_argument('--refs', '-r', nargs='+')
    parser.add_argument('-n', type=int, default=4)
    args = parser.parse_args()

    main(args)
