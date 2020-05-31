#!/bin/sh

thisdir=$(dirname $0)
. $thisdir/training-vars

model=$1
beamsize=$2
GPUS="$3"


$marian/build/marian-decoder -m $model \
			     --devices $GPUS \
			     -v $vocab_joint $vocab_joint  \
			     -b $beamsize --normalize 0.6 \
			     --n-best \
			     --mini-batch 8 --maxi-batch-sort src --maxi-batch 100 -w 2500


