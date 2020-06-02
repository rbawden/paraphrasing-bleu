#!/usr/bin/python3

"""
uniq that only compares first two tab-delimited fields.
"""

import sys

prev_key = None
for line in sys.stdin:
    tokens = line.split('\t')
    key = '\t'.join(tokens[:2])
    if prev_key == key:
        continue
    else:
        prev_key = key
        print(line, end='')
