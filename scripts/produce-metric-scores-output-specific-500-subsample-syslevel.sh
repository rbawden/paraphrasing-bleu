#!/bin/bash

# Command line arguments
eval_tool=$1        # Eval script in scripts/ (either sacreBLEU-truncate-syslevel.sh or Meteor-truncate-syslevel.sh)
name=$2             # Specific name of the metric (to be written into the output file)
model_type=$3       # "laser" or "treelstm"

# Check command line arguments
if [[ "$#" -ne 3 ]]; then
    echo "Wrong number of parameters given"
    echo "Usage: $0 EVAL_TOOL NAME MODEL_TYPE"
    echo -e "\tEVAL_TOOL: path to eval script in scripts/"
    echo -e "\tNAME: the metric's name"
    echo -e "\tMODEL_TYPE: laser or treelstm"
    exit
fi

testset=newstest2019

# Specify folder containing all system inputs
thisdir=`dirname $0`
projdir=$thisdir/..
refdir="$projdir/original-references/" # reference files store in here
hypdir="$projdir/mt_submissions/$testset"
paradir="$projdir/paraphrases/$testset/mt-specific-outputs/"

# for only language pair
for lpnodash in deen; do
    # modified version with dash (needed for system submissions)
    lp=`echo $lpnodash | perl -pe 's/(..)(..)/\1-\2/'`
    
    # Skip language pair if target not English
    if [[ $lp != *en ]]; then
	continue
	fi
    # Get reference
    ref=`ls $refdir/$testset/*$lpnodash-ref.en.500`
    
    # For each set of hypotheses, compute the metric score
    for hyp in $hypdir/$testset*$lp; do
	
	systemname=`echo $hyp | perl -pe "s/^.+?\/\$testset\.(.+?\.\d+)\.\$lp$/\1/"`
	# Get specific paraphrase
	para=`echo $paradir/$testset.$systemname.$lp.*$model_type*`
	randnum=$RANDOM
	cat $para | head -n 500 > $paradir/$randnum
	score=`$eval_tool $hyp "500" "$ref $paradir/$randnum" 1`
	echo -ne "$name\t$lp\t$testset\t$systemname\t$score\n"
	rm $paradir/$randnum
    done
done


