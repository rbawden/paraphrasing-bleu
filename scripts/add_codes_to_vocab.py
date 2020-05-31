#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_vocab')
parser.add_argument('num_codes', type=int, help='number of codes to add (they will start at 0)')
args = parser.parse_args()

last_id = 0
with open(args.yaml_vocab) as fp:
    for line in fp:
        last_id = int(line.strip().rsplit(':')[-1])

# now append the clusters to the end of the vocab
with open(args.yaml_vocab, 'a') as fp:
    fp.write('\n')
    for i in range(args.num_codes):
        fp.write('<cl' + str(i) + '>: ' + str(last_id + i + 1) + '\n')
