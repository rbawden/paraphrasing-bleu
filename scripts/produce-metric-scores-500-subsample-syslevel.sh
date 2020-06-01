#!/bin/bash

# Command line arguments
eval_tool=$1        # Eval script in scripts/ (either sacreBLEU-truncate-syslevel.sh or Meteor-truncate-syslevel.sh)
metricname=$2       # Specific name of the metric (to be written into the output file)
para_ref_folder=$3  # Folder of additional paraphrased refs (if not just using the original reference)
n=$4                # The number of additional paraphrased references to use (not including the real reference)

# Check command line arguments
if [[ "$#" -ne 4 && "$#" -ne 2 ]]; then
    echo "Wrong number of parameters given"
    echo "Usage: $0 EVAL_TOOL NAME [PARA_REF_FOLDER] [N]"
    echo -e "\tEVAL_TOOL: path to eval script in scripts/"
    echo -e "\tNAME: the metric's name"
    echo -e "\tPARA_REF_FOLDER (optional): path to folder containing paraphrased references. If not given, only original re\
ferences are used."
    echo -e "\tN (optional, if PARA_REF_FOLDER specified): the number of paraphrased references to use"
    exit
fi


if [ -z $n ]; then
    n=0
fi

# Specify folder containing all system inputs
thisdir=`dirname $0`
refdir="$thisdir/../original-references/" # reference files store in here
hypdir="$thisdir/../mt_submissions/newstest2019"
human_para_folder=$thisdir/../paraphrases/newstest2019/human
testset=newstest2019

# get number of lines of human paraphrases (this is where the paraphrase files will be truncated too)
num_lines=`wc -l $human_para_folder/deen-1.en.500 | perl -pe 's/ *(\d+).+?$/\1/'`


# for each language pair
for lpnodash in deen; do
    # modified version with dash (needed for system submissions)
    lp=de-en
    
    # Use single references if no ref_folder specified, otherwise use all files in ref_folder containing $lp
    if [ -z $para_ref_folder ]; then
	ref="$refdir/$testset/$testset-$lpnodash-ref.en.$num_lines"
    else
	ref=`ls $para_ref_folder/$lpnodash-*.en.$num_lines`
	
	# if $n is specified, then only select reference files with names up to $n
	if [ ! -z $n ]; then
	    ref=`ls $para_ref_folder/$lpnodash-*.en.$num_lines | perl -pe 's/\-(\d+)\./°\1°/g' | awk -F"°" -v var="$n" '{if ($2 <= var) print}' | perl -pe 's/°/-/g; s/\-en\./.en./'`
	fi
	# always add main reference file
	ref+=" $refdir/$testset/$testset-$lpnodash-ref.en.$num_lines"
    fi
    
    # For each set of hypotheses, compute the metric score
    for hyp in $hypdir/$testset*$lp; do
	systemname=`echo $hyp | perl -pe "s/^.+?\/\$testset\.(.+?\.\d+)\.\$lp$/\1/"`
	score=`$eval_tool $hyp "$num_lines" "$ref" $n`
	echo -ne "$metricname\t$lp\t$testset\t$systemname\t$score\n"
    done
done

