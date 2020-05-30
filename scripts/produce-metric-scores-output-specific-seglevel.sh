#!/bin/bash

# Command line arguments
testset=$1
modeltype=$2
eval_tool=$3

# Check command line arguments
if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 TESTSET MODELTYPE EVALTOOL"
    echo -e "\tTESTSET: newstest2019 or newstest2018"
    echo -e "\tMODELTYPE: laser or treelstm"
    echo -e "\tEVALTOOL: scripts/sacreBLEU-seglevel or scripts/Meteor-seglevel"
    exit
fi

# Specify folder containing all system inputs
thisdir=`dirname $0`
projdir=$thisdir/..
refdir="$projdir/original-references/" # reference files store in here
hypdir="$projdir/mt_submissions/$testset"
paradir="$projdir/paraphrases/$testset/mt-specific-outputs/"

# for each language pair
for lpnodash in `ls $refdir/$testset/*-ref.en | perl -pe 's/^.+\-([^-]+?en)\-ref\.en$/\1/'`; do

    # modified version with dash (needed for system submissions)
    lp=`echo $lpnodash | perl -pe 's/(..)(..)/\1-\2/'`
    
    # Skip language pair if target not English
    if [[ $lp != *en ]]; then
	continue
	fi
    # Get reference
    ref=`ls $refdir/$testset/*$lpnodash-ref.en`
    
    # For each set of hypotheses, compute the metric score
    for hyp in $hypdir/$testset*$lp; do
	
	systemname=`echo $hyp | perl -pe "s/^.+\/\$testset\.([^.]+?\.\d+)\.\$lp$/\1/"`
	# Get specific paraphrase
	para=`echo $paradir/$testset.$systemname.$lp.*$modeltype*`

	scores=`$eval_tool $hyp "$ref $para" 1`
	i=1
	for score in $scores; do
	    echo -ne "$modeltype\t$lp\t$testset\t$systemname\t$i\t$score\n"
	    i=$((i+1))
	done
    done
done


