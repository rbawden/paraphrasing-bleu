#!/bin/bash

#$ -S /bin/bash -cwd
#$ -j y -o logs
#$ -l h_rt=24:00:00

. ~/.bashrc

set -eu

input=$1

split -l 1000 --numeric-suffixes=1 -a 3 $input input.

ls input.* > pieces.txt
numlines=$(cat pieces.txt | wc -l)

qsub -sync y -t 1:$numlines /home/hltcoe/mpost/code/sentcodes/scripts/translate-piece.sh pieces.txt

cat out.input.* > out.$(basename $input)
ln -sf out.$(basename $input) translations
