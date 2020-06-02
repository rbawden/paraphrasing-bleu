#!/bin/bash

set -eu

infile=$1
outfile=out.$(basename $infile)

[[ -s $outfile ]] && exit

cat $infile \
  | PYTHONPATH=~/code/sockeye-trie/ python3 -m sockeye.translate \
    -m model \
    --device-ids 0 --disable-device-locking \
    --max-input-len 200 \
    --json-input --beam-size 20 --batch-size 20 --beam-prune 30 --output-type json \
    > $outfile
