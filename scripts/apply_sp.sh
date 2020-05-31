sentencepiece=$1
model=$2

$sentencepiece/build/src/spm_encode --model $model
