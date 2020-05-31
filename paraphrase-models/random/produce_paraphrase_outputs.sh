#!/bin/sh

# Command line arguments
model=$1   # path to model.npz file
n=$2       # beam size
testset=$3
gpus="$4"  # GPUS of format "0 1 2"

# Check arguments
if [ "$#" -ne 4 ]; then
    echo -e "Erorr: Wrong number of arguments given (expecting 3)."
    echo -e "Usage: $0 MODEL BEAMSIZE GPUS"
    exit
fi

# Paths, variables, inputs and outputs
thisdir=$(dirname $0)
. $thisdir/training-vars

output_folder=$projdir/paraphrases/${testset}/random

# Put output files in this folder
input_prefix=$projdir/data/${testset}/${testset} # Folder of references of name newstest2019-$lp-ref.sp.en
mkdir -p $output_folder/

beam=6

# For each of the language pairs into English
for lp in csen deen eten fien guen kken lten ruen tren zhen; do

    if [ ! -f input=$input_prefix-$lp-ref.norm.sp.en ]; then
	continue
    fi
    
    # get input and output paths
    input=$input_prefix-$lp-ref.norm.sp.en
    echo "Translating $input with beam size $n and outputting to $output_folder/$lp-{1..$n}.en"
    # first produce results with beam size $n
    # decode from codes 1 to n
    for i in $(seq 1 $n); do
	cat $input | bash $thisdir/translate.sh $model $beam $i "$gpus" > $output_folder/$lp-$i.en
	# postprocess
	cat $output_folder/$lp-$i.en | perl -pe 's/^<cl[0-9]+> *//' | perl -pe 's/ *//g' | perl -pe 's/▁/ /g' \
								 > $output_folder/$lp-$i.postproc
	# move the file so that there is only one version in the folder
	mv $output_folder/$lp-$i.postproc $output_folder/$lp-$i.en
    done
    
done
