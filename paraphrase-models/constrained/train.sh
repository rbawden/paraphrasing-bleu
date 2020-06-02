#!/bin/bash

export dev_src="/path/to/data/dev.src"
export dev_trg="/path/to/data/dev.trg"
export prepared_data="/path/to/prepared/data"
export model="model"
export pyenv="conda:sockeye10"
export transformer_attention_heads="8"
export submitter="sge"
export train_batch_size="10000"
export transformer_feed_forward_num_hidden="2048"
export train_checkpoint_freq="5000"
export num_layers="6:6"
export decoder_type="transformer"
export num_embed="512:512"
export train_num_decode_and_eval="500"
export transformer_model_size="512"
export source_factors_combine="sum"
export num_devices="4"
export encoder_type="transformer"
export train_batch_type="word"
export train_max_checkpoints_not_improved="10"
export source_factors_num_embed="512 512"
export sockeye="/path/to/sockeye-trie"

export PYTHONPATH=${sockeye}
 python3 -m sockeye.train \
    -o $model \
    --device-ids -$num_devices
    --disable-device-locking \
    --prepared-data $prepared_data \
    --num-embed=$num_embed \
    --validation-source $dev_src \
    --validation-target $dev_trg \
    --encoder=$encoder_type \
    --decoder=$decoder_type \
    --num-layers=$num_layers \
    --transformer-model-size=$transformer_model_size \
    --transformer-attention-heads=$transformer_attention_heads \
    --transformer-feed-forward-num-hidden=$transformer_feed_forward_num_hidden \
    --transformer-positional-embedding-type=fixed \
    --transformer-preprocess=n \
    --transformer-postprocess=dr \
    --transformer-dropout-attention=0.1 \
    --transformer-dropout-act=0.1 \
    --transformer-dropout-prepost=0.1 \
    --embed-dropout=0.1:0.1 \
    --weight-tying \
    --weight-tying-type=src_trg_softmax \
    --weight-init=xavier \
    --weight-init-scale=3.0 \
    --weight-init-xavier-factor-type=avg \
    --optimizer=adam \
    --optimized-metric=perplexity \
    --label-smoothing=0.1 \
    --gradient-clipping-threshold=-1 \
    --initial-learning-rate=0.0002 \
    --learning-rate-reduce-num-not-improved=8 \
    --learning-rate-reduce-factor=0.9 \
    --learning-rate-scheduler-type=plateau-reduce \
    --learning-rate-decay-optimizer-states-reset=best \
    --learning-rate-decay-param-reset \
    --max-num-checkpoint-not-improved $train_max_checkpoints_not_improved \
    --batch-type=$train_batch_type \
    --batch-size=$train_batch_size \
    --checkpoint-interval=$train_checkpoint_freq \
    --decode-and-evaluate=$train_num_decode_and_eval \
    --decode-and-evaluate-device-id -1 \
    --keep-last-params=10

