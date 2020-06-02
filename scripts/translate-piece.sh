#!/bin/bash

#$ -S /bin/bash -cwd
#$ -j y -o logs
#$ -q gpu.q@@2080 -l h_rt=24:00:00,gpu=1

. ~/.bashrc

set -eu

echo HOSTNAME
hostname
echo

nvidia-smi
echo

echo ENV
env | grep SGE
env | grep CUDA
echo

conda deactivate
conda activate sockeye10

infile=$(mid $SGE_TASK_ID $1)
outfile=out.$(basename $infile)

[[ -s $outfile ]] && exit

cat $infile \
  | PYTHONPATH=~/code/sockeye-trie/ python3 -m sockeye.translate \
    -m /exp/mpost/parbleu19/constraints/model \
    --device-ids 0 --disable-device-locking \
    --max-input-len 200 \
    --json-input --beam-size 20 --batch-size 20 --beam-prune 30 --output-type json \
    > $outfile
