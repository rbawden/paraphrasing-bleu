#!/bin/bash

# Command line arguments
testset=$1          # Either "newstest2018" or "newstest2019"
eval_tool=$2        # Eval script in scripts/ (either sacreBLEU-syslevel.sh or Meteor-syslevel.sh)
metricname=$3       # Specific name of the metric (to be written into the output file)
para_ref_folder=$4  # Folder of additional paraphrased refs (if not just using the original reference)
                    #     Within this folder, reference files should be named $lp-$i,
                    #     where $lp is language pair (e.g deen, fien) and $i is the additional reference num)
n=$5                # The number of additional paraphrased references to use (not including the real reference)

# Check command line arguments
if [[ "$#" -ne 5 && "$#" -ne 3 ]]; then
    echo "Wrong number of parameters given"
    echo "Usage: $0 TESTSET EVAL_TOOL NAME [PARA_REF_FOLDER] [N]"
    echo -e "\tTESTSET: Either 'newstest2018' or newstest2019'"
    echo -e "\tEVAL_TOOL: path to eval script in scripts/"
    echo -e "\tNAME: the metric's name"
    echo -e "\tPARA_REF_FOLDER (optional): path to folder containing paraphrased references. If not given, only original references are used."
    echo -e "\tN (optional, if PARA_REF_FOLDER specified): the number of paraphrased references to use"
    exit
fi


# Folders containing original references and MT submissions to be scored
thisdir=`dirname $0`
refdir="$thisdir/../original-references/" # reference files stored in here
hypdir="$thisdir/../mt_submissions/"


# for each language pair (no dash, e.g. deen, fien)
for lpnodash in `ls $refdir/$testset/*-ref.en | perl -pe 's/^.+\-([^-]+?en)\-ref\.en$/\1/'`; do
    
    # modified version with dash (needed for system submissions)
    lp=`echo $lpnodash | perl -pe 's/(..)(..)/\1-\2/'`
    
    # Skip language pair if target not English
    if [[ $lp != *en ]]; then
	continue
    fi
    
    # Use single references if no para_ref_folder specified, otherwise use all files in para_ref_folder containing $lp
    if [ -z $para_ref_folder ]; then
	ref="$refdir/$testset/$testset-$lpnodash-ref.en"
    else
	# all paraphrased references
	ref=`ls $para_ref_folder/*$lpnodash-*.en`

	# if $n is specified, then only select reference files with names up to $n
	if [ ! -z $n ]; then
	    ref=`ls $para_ref_folder/$lpnodash-*.en | perl -pe 's/\-(\d+)\./°\1°/g' | awk -F"°" -v var="$n" '{if ($2 <= var) print}' | perl -pe 's/°/-/g; s/\-en$/.en/'`
	fi
	# always add main reference file
	ref+=" $refdir/$testset/$testset-$lpnodash-ref.en"
    fi
    
    # For each set of hypotheses, compute the metric score
    for hyp in $hypdir/$testset/$testset*$lp; do
	systemname=`echo $hyp | perl -pe "s/^.+?\/\$testset\.(.+?\.\d+)\.\$lp$/\1/"`
	score=`bash $eval_tool $hyp "$ref" $n` # include number for meteor, ignored by sacrebleu script
	echo -ne "$metricname\t$lp\t$testset\t$systemname\t$score\n"
    done
done

