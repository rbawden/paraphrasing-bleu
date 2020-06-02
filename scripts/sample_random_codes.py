#!/usr/bin/python
import argparse
import random

argparse = argparse.ArgumentParser()
argparse.add_argument('numsents', type=int,
                      help='Total number of codes to generate')
argparse.add_argument('--code_range', nargs=2, type=int, required=True,
                      help='Id numbers of first and last codes')
args = argparse.parse_args()

# Use this when you want randomly sampled codes from
# args.code_range[0] to args.code_range[1] (included)
# N.B. you want also want to take codes generated using
# another method and shuffle them (in order to maintain
# the same code distribution)


for _ in range(args.numsents):
    num = random.randint(args.code_range[0],args.code_range[1])
    print('<cl' + str(num) + '>')
