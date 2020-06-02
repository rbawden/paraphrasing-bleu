#!/usr/bin/python3

# for pruning parabank 2, takes only the first unique target line, fifthing the data
# optionally take the most different one

import sys

def dice(sent1, sent2):
    set1 = set(sent1.split())
    set2 = set(sent2.split())
    return 2 * len(set1.intersection(set2)) / (len(set1) + len(set2))

def main(args):

    last_trg = None
    choice = [1, None]
    for line in sys.stdin:
        src, trg = line.rstrip().split('\t')

        if trg != last_trg:
            if last_trg != None:
                print(choice[1], last_trg, sep='\t')
                choice = [1, None]

        if args.method == 'diff':
            score = dice(src, trg)
            if score < choice[0]:
                choice = [score, src]
        else:
            if choice[1] == None:
                choice = [1, src]

        last_trg = trg

    print(choice[1], last_trg, sep='\t')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['first', 'diff'], default='first')
    args = parser.parse_args()

    main(args)
