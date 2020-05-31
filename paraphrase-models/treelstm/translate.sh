#!/bin/sh

thisdir=$(dirname $0)
. $thisdir/training-vars

model=$1
beamsize=$2
nth=$3
GPUS="$4"


$marian/build/marian-decoder -m $model \
			     --devices $GPUS \
			     -v $vocab_joint $vocab_joint  \
			     -b $beamsize --normalize 0.6 \
			     --nth-code $nth \
			     --mini-batch 8 --maxi-batch-sort src --maxi-batch 100 -w 2500


