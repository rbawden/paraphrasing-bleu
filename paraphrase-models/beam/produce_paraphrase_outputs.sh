#!/bin/sh

# Command line arguments
model=$1   # path to model.npz file
n=$2       # beam size (i.e. number of paraphrases to produce)
testset=$3 # newstest2018 or newstest2019
gpus="$4"  # GPUS of format "0 1 2"

# Check arguments
if [ "$#" -ne 4 ]; then
    echo -e "Erorr: Wrong number of arguments given (expecting 3)."
    echo -e "Usage: $0 MODEL NUM_PARAS TESTSET GPUS"
    exit
fi

# Paths, variables, inputs and outputs
thisdir=$(dirname $0)
. $thisdir/training-vars

output_folder=$projdir/paraphrases/${testset}/beam # Put output files in this folder
input_folder=$projdir/original-references/${testset} # Folder of references of name newstest2019-$lp-ref.sp.en
mkdir -p $output_folder/orig_files


# For each of the language pairs into English (from 18 or 19)
for lp in csen deen eten fien guen kken lten ruen tren zhen; do

    # if this is a relevant language for this test set
    if [ ! -f $input_folder/${testset}-$lp-ref.norm.sp.en ]; then
	continue
    fi
    
    # Translate with beam size $n and output into a single file per language
    if [ ! -f $output_folder/orig_files/$lp-tmp-$n ]; then
	input=$input_folder/${testset}-$lp-ref.norm.sp.en
	echo "Translating $input with beam size $n and outputting to $output_folder/orig_files/$lp-tmp-$n"
	# first produce results with beam size $n
	cat $input | bash $thisdir/translate.sh $model $n "$gpus" > $output_folder/orig_files/$lp-tmp-$n
    else
	echo "$output_folder/orig_files/$lp-tmp-$n already exists. First delete if you want to redo translation."
    fi

    # Split file into several files for each $i from 1..$beam
    # Make sure files are empty first
    for i in $(seq 1 $n); do
	if [ -f $output_folder/$lp-$i ]; then
	    echo "$output_folder/$lp-$i already exists. Please delete first if you want to proceed."
	fi
    done

    # Write to files
    i=0 # line num
    while IFS= read -r line; do
	file_num=$(($i % $n + 1))
	echo "$line" | cut -d"|" -f 4 >> $output_folder/$lp-$file_num
	i=$((i+1))
    done < $output_folder/orig_files/$lp-tmp-$n

    # Postprocess files with sentencepiece
    for i in $(seq 1 $n); do
	cat $output_folder/$lp-$i | \
	    perl -pe 's/^<cl[0-9]+> *//' | perl -pe 's/ *//g' | perl -pe 's/▁/ /g' \
								     > $output_folder/$lp-$i.postproc
	# Remove processed files to avoid confusion and rename postproc file
	rm $output_folder/$lp-$i
	mv $output_folder/$lp-$i.postproc $output_folder/$lp-$i.en
    done
done


