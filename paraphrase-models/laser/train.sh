#!/bin/bash

GPUS=$@

# load training variables
thisdir=$(dirname $0)
. $thisdir/training-vars

# make model if it does not exist
mkdir -p $modeldir

# train!
$marian/build/marian \
    --devices $GPUS \
    --model $modeldir/model.npz \
    --type transformer \
    --train-sets $train_src $train_trg \
    --max-length 120 \
    --vocabs $vocab_joint $vocab_joint \
    --mini-batch-fit \
    -w 6000 \
    --maxi-batch 500 \
    --early-stopping 10 \
    --cost-type ce-mean-words \
    --valid-freq 5000 \
    --save-freq 5000 \
    --disp-freq 500 \
    --log $modeldir/train.log \
    --valid-log $modeldir/valid.log \
    --valid-metrics ce-mean-words perplexity translation \
    --valid-sets $dev_src $dev_trg \
    --valid-translation-output $modeldir/valid.output \
    --valid-mini-batch 16 \
    --valid-max-length 1000 \
    --beam-size 6 \
    --normalize 0.6 \
    --enc-depth 6 \
    --dec-depth 6 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 \
    --tied-embeddings-all \
    --label-smoothing 0.1 \
    --learn-rate 0.0003 \
    --lr-warmup 16000 \
    --lr-decay-inv-sqrt 16000 \
    --lr-report \
    --optimizer-params 0.9 0.98 1e-09 \
    --clip-norm 5 \
    --sync-sgd \
    --exponential-smoothing \
    --valid-script-path $maindir/validate.sh \
    --optimizer-delay 2
