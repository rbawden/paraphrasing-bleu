#!/usr/bin/env python3

import glob
import os
import html
import sys
import random
import csv
import itertools

from collections import defaultdict


# Input directory
inp_dir = sys.argv[1]
out_file = sys.argv[2]

# number of sentences
n = int(sys.argv[3])

# fix seed if wanted
if len(sys.argv) > 4:
    np_seed = int(sys.argv[4])
else:
    np_seed = 21


ref_file = glob.glob(f"{inp_dir}/references/*")[0]
refs = []
with open(ref_file) as fin:
    for idx, line in enumerate(fin):
        if idx == n:
            break
        refs.append(html.escape(line.strip()))

dd = [defaultdict(list) for i in range(n)]
for i, filename in enumerate(glob.iglob(f"{inp_dir}/*/*")):
    if not "/references/" in filename:
        _, system, basename = filename.replace(".en", "").split("/")
        f_name = f"{system}-{basename}"
        with open(filename) as fin:
            for idx, line in enumerate(fin):
                if idx == n:
                    break
                dd[idx][system].append((f_name, refs[idx], html.escape(line.strip()), idx))


num_hits = len(dd[0].keys())
with open(out_file, encoding="utf8", mode="w") as fout:
    tsv_writer = csv.writer(fout, delimiter=',', lineterminator='\n')

    prep_list = [(f"sysname_{i}", f"ref_{i}", f"sys_{i}", f"selected_{i}") for i in range(n)]
    prep_list = list(itertools.chain.from_iterable(zip(*prep_list)))
    tsv_writer.writerow(prep_list)
    for hit in range(num_hits):
        selections = []
        for i in range(n):
            system = random.choice(list(dd[i].keys()))
            which = random.randrange(len(dd[i][system]))
            selections.append(dd[i][system].pop(which))
            print(f"POP {i} {system} {which} {len(dd[i][system])}", file=sys.stderr)
            if len(dd[i][system]) == 0:
                del dd[i][system]

        prep_list = list(itertools.chain.from_iterable(zip(*selections)))
        tsv_writer.writerow(prep_list)
