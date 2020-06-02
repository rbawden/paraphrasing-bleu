#!/usr/bin/env python3

"""
Combines TSV files output by find_new_ngrams.py (so you can look across languages)

e.g.,

    ../scripts/find_new_ngrams_combine.py -n 4 counts.{de,fi,gu,kk,lt,ru,zh} --max 10

"""

import sys
import csv

from collections import defaultdict
from operator import itemgetter

def main(args):
    csv.field_size_limit(sys.maxsize)

    counts = defaultdict(int)
    for file in args.files:
        for row in csv.DictReader(file, delimiter="\t"):
            # pair    N       count   ngram   lines
            ngram = row["ngram"]
            if args.n and len(ngram.split()) != args.n:
                continue

            count = int(row["count"])
            counts[ngram] += count

    for i, (ngram, count) in enumerate(sorted(counts.items(), key=itemgetter(1), reverse=True)):
        if args.max and i > args.max:
            break
        print(f"{ngram} ({count}) ", end="")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=argparse.FileType("r"))
    parser.add_argument("-n", type=int)
    parser.add_argument("--max", type=int)
    args = parser.parse_args()

    main(args)
