#!/usr/bin/env python3

"""
Takes human scores, system outputs, and one or more references.
Counts new ngrams appearing in refs 2+ and *not* in ref #1.

Example usage:

for lang in de fi gu kk lt ru zh; do 
  $HOME/code/sentcodes/scripts/find_new_ngrams.py -s $HOME/exp/parbleu19/paraphrases/newstest2019/systems/$lang-en/newstest2019.*.$lang-en -r $HOME/exp/parbleu19/paraphrases/newstest2019/references/wmt19.$lang-en.en $HOME/exp/parbleu19/paraphrases/newstest2019/treelstm-plain-545000/${lang}en-{1,2,3,4,5}.en --human-scores $HOME/exp/parbleu19/paraphrases/newstest2019/scores.human.$lang --human-min 0 --human-max 100 -m 5 -n 5 > counts.$lang; 
done
"""

import os
import sacrebleu
import sys

from collections import Counter, defaultdict
from operator import itemgetter

def main(args):

    if args.human_scores:
        systems = []
        scores = {}
        for line in open(args.human_scores):
            system, score = line.rstrip().split()
            scores[system] = float(score)
        for system_path in args.systems:
            system_name = '.'.join(os.path.basename(system_path).split('.')[1:-1])
            if system_name not in scores:
                print(f"COULDN'T FIND SYSTEM {system_name} ({system_path})", file=sys.stderr)
            elif scores[system_name] >= args.human_min and scores[system_name] <= args.human_max:
                systems.append(system_path)
    else:
        systems = args.systems

    print(f"SYSTEMS[{args.human_min} to {args.human_max}]: {systems}", file=sys.stderr)

    langpair = os.path.basename(args.refs[0]).split(".")[1]

    sys_fhs = [open(ref) for ref in systems]
    ref_fhs = [open(system) for system in args.refs]

    tokenize = sacrebleu.TOKENIZERS["13a"]

    new_ngrams = Counter()
    new_ngram_data = defaultdict(list)
    totals = Counter()
    for lineno, (syss, refs) in enumerate(zip(zip(*sys_fhs), zip(*ref_fhs)), 1):
        syss = [tokenize(system) for system in syss]
        refs = [tokenize(ref) for ref in refs]

        # All system n-grams.
        sys_ngrams = Counter()
        for system in syss:
            sys_ngrams += sacrebleu.extract_ngrams(system, min_order=args.m, max_order=args.n)

        # reset counts for all n-grams found in references
        ref_ngrams = Counter()
        first_ref_ngrams = None
        for ref in refs:
            all_ngrams = sacrebleu.extract_ngrams(ref, min_order=args.m, max_order=args.n)

            # Only keep ngrams that don't appear in longer ngrams
            used_ngrams = []
            keep_ngrams = {}
            if args.uniq:
                for ngram in sorted(all_ngrams, key=len, reverse=True):
                    for used in used_ngrams:
                        if ngram in used:
                            break
                    else:
                        keep_ngrams[ngram] = all_ngrams[ngram]
                        used_ngrams.append(ngram)
            else:
                keep_ngrams = all_ngrams

            if not first_ref_ngrams:
                first_ref_ngrams = keep_ngrams
            else:
                ref_ngrams += keep_ngrams

        for ngram in sys_ngrams.keys():
            totals[ngram] += 1

            if ngram in ref_ngrams and ngram not in first_ref_ngrams:
                new_ngrams[ngram] += 1
                new_ngram_data[ngram].append(lineno)

    print("pair", "N", "count", "ngram", "lines", sep="\t")
    for ngram, count in sorted(new_ngrams.items(), key=lambda x: (len(x[0].split()), x[1]), reverse=True):
        lines = " ".join(map(str, new_ngram_data[ngram]))
        print(langpair, len(ngram.split()), count, ngram, lines, sep="\t")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', '-s', nargs='+')
    parser.add_argument('--refs', '-r', nargs='+')
    parser.add_argument('--human-scores', help='file with human scores')
    parser.add_argument('--human-min', "-hm", type=int, default=0, help="human score minimum")
    parser.add_argument('--human-max', "-hn", type=int, default=100, help="human score maximum")
    parser.add_argument('--uniq', "-u", action="store_true", help="only unique ngrams")
    parser.add_argument('-m', type=int, default=1)
    parser.add_argument('-n', type=int, default=4)
    args = parser.parse_args()

    main(args)
