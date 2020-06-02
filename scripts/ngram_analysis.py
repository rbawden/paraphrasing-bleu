#!/usr/bin/env python3

"""
This script computes the ngram coverage of a single reference against an additional set.
It can optionally limit coverage to just those systems that are highly ranked by humans.

    ngram_analysis.py \
      wmt18-submitted-data/txt/references/newstest2018-deen-ref.en \
      wmt18-submitted-data/txt/system-outputs/newstest2018/de-en/newstest2018.* \
      -t 3 --normalize --spm /path/to/spm/model
"""

import argparse
import json
import sacrebleu
import os
import sys
import sentencepiece as spm
import sacremoses

from sacremoses.normalize import MosesPunctNormalizer
from collections import defaultdict, Counter


def main(args):

    print(args, file=sys.stderr)

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
            elif scores[system_name] > args.scope:
                systems.append(system)
    else:
        systems = args.systems

    if args.normalize:
        normalizer = MosesPunctNormalizer(lang='en', penn=False)

    if args.spm:
        sp = spm.SentencePieceProcessor()
        sp.Load(args.spm)


    fds = [open(file) for file in systems]

    num_constraints = 0
    num_skipped = 0
    for lineno, (ref, *systems) in enumerate(zip(open(args.reference), *fds), 1):

        def preprocess(text):
            if args.normalize:
                text = normalizer.normalize(text)
            if args.spm:
                text = ' '.join(sp.EncodeAsPieces(text))
            return ' '.join(text.split()[:args.maxlen])

        if len(ref.split()) > args.maxlen:
            continue

        ref_ngrams = sacrebleu.extract_ngrams(ref, min_order=args.ngram_min, max_order=args.ngram_max)

        paraphrase_ngrams = Counter()
        for paraphrase in args.paraphrases:
            paraphrase_ngrams += sacrebleu.extract_ngrams(paraphrase, min_order=args.ngram_min, max_order=args.ngram_max)

        ngrams = Counter()
        for system in systems:
            ngrams += sacrebleu.extract_ngrams(system, min_order=args.ngram_min, max_order=args.ngram_max)

        for ngram in ref_ngrams.keys():
            ngrams[ngram] = 0
        ngrams -= ref_ngrams
        if args.threshold <= 1:
            attested_ngrams = [ngram for ngram in ngrams.keys() if (ngrams[ngram] / len(systems)) >= args.threshold]
        else:
            attested_ngrams = [ngram for ngram in ngrams.keys() if ngrams[ngram] >= args.threshold]

        used_ngrams = []
        for ngram in sorted(attested_ngrams, key=len, reverse=True):
            for used in used_ngrams:
                if ngram in used:
#                    print(f"** {lineno} already saw '{ngram}' in '{used}', skipping", file=sys.stderr)
                    num_skipped += 1
                    break
            else:
                num_constraints += 1
                used_ngrams.append(ngram)
                j = { 'sentno': lineno,
                      'text': preprocess(ref),
                      'constraints': [preprocess(ngram)] }
                print(json.dumps(j, ensure_ascii=False), flush=True)
        #print(*attested_ngrams, sep='\t', flush=True)

    print(f"Created {num_constraints} constrained sentences, skipping {num_skipped} smaller ones", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--paraphrases', '-p', nargs='+')
    parser.add_argument('--systems', '-s', nargs='+')
    parser.add_argument('--human-scores', help='file with human scores')
    parser.add_argument('--maxlen', '-m', type=int, default=80)
    parser.add_argument('--spm', help="Path to sentencepiece model")
    parser.add_argument('--normalize', '-n', action='store_true', help="Apply Moses normalization")
    parser.add_argument('--ngram-min', type=int, default=1)
    parser.add_argument('--ngram-max', type=int, default=4)
    args = parser.parse_args()

    main(args)
