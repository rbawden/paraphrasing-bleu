#!/bin/bash

thisdir=$(dirname $0)
. $thisdir/training-vars
source $projdir/pyenv/bin/activate

export LC_ALL=C.UTF-8

# remove code and postprocess
cat $1 | perl -pe 's/^<cl[0-9]+> *//' | perl -pe 's/ *//g' | perl -pe 's/▁/ /g' > $modeldir/tmp.postproc

# calculate BLEU
cat $modeldir/tmp.postproc | sacrebleu $ref | perl -pe 's/BLEU.*? = ([0-9.]+) .*/\1/'

