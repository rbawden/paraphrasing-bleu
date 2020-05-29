#!/bin/sh

testset=$1
level=$2
metric=$3 # bleu or meteor

# Check command line arguments
if [[ "$#" -ne 3 ]]; then
    echo "Wrong number of parameters given"
    echo "Usage: $0 TESTSET LEVEL NAME METRIC"
    echo -e "\tTESTSET: Either 'newstest2018' or newstest2019'"
    echo -e "\tLEVEL: 'sys' or 'seg'"
    echo -e "\tMETRIC: 'bleu' or 'meteor'"
    exit
fi

thisdir=`dirname $0`
outputs=$thisdir/../metric-scores/$testset
metricstask=$thisdir/../metrics-task
scriptdir=$thisdir/../scripts


# Get baseline scripts for the chosen metric
if [ $metric == 'bleu' ]; then
    baseline=$outputs/sacreBLEU-${level}level.tsv
elif [ $metric == 'meteor' ]; then
    baseline=$outputs/Meteor-${level}level.tsv
else
    echo "Please choose a valid metric"
    exit
fi

# Get gold file
if [ $level == 'seg' ]; then
    gold=$metricstask/RR-seglevel-$testset.csv
else
    gold=$metricstask/DA-syslevel-$testset.csv
fi

# Calculate the correlation
function correlate {
    python $scriptdir/metric_correlation-${level}level.py $gold $baseline $1 --just_scores
}



# Headers and languages
echo "\\centering\small
\\scalebox{1}{
\\begin{tabular}{lllllllll}
\\toprule"

langs=`cut -f 2 $outputs/beam/par${metric}-beam.num=5-${level}level.tsv | sort -u | perl -pe 's/\s/ \& /g'`
echo "&& $langs"
echo " Approach & Method \\"


# Baseline results
if [ $level == 'seg' ]; then
    if [ $metric == 'bleu' ]; then
	printf "Baseline & sentence\\\bleu "
    else
	 printf "Baseline & sentence\\\meteor "
    fi
else
    if [ $metric == 'bleu' ]; then
	printf "Baseline & \\\bleu "
    else
	printf "Baseline & \\\meteor "
    fi
fi
correlate $baseline

# Paraphrase baseline +5 models
echo -e "\n\\midrule"
echo "\\multirow{3}{*}{\pbox{1.5cm}{Paraphrase baselines (+5)}}"

printf "& \\\beam "
correlate $outputs/beam/par${metric}-beam.num=5-${level}level.tsv

printf "& \\\random "
correlate $outputs/random/par${metric}-random.num=5-${level}level.tsv

printf "& \\\sampled "
if [ -f $outputs/sampled/par${metric}-sampled.num=5-${level}level.tsv ]; then
    correlate $outputs/sampled/par${metric}-sampled.num=5-${level}level.tsv
fi

# Paraphrase +1 diverse models
echo -e "\n\\midrule"
echo "\\multirow{3}{*}{\\pbox{1.3cm}{Diversity (+1)}} "

printf "& \\\laser "
correlate $outputs/laser/par${metric}-laser.num=1-${level}level.tsv

printf "& \\\treelstm "
correlate $outputs/treelstm/par${metric}-treelstm.num=1-${level}level.tsv

# Paraphrase +5 diverse models
echo -e "\n\\midrule"
echo "\\multirow{3}{*}{\\pbox{1.3cm}{Diversity (+5)}} "

printf "& \\\laser "
correlate $outputs/laser/par${metric}-laser.num=5-${level}level.tsv

printf "& \\\treelstmplain "
correlate $outputs/treelstm/par${metric}-treelstm.num=5-${level}level.tsv


# Output-specific (+1) models
echo -e "\n\\midrule"
echo "\\multirow{2}{*}{\\pbox{1.3cm}{Output-specific (+1)}}"

printf "& \\\laser "
correlate $outputs/mt-output-specific/par${metric}-constrained-laser.${level}level.tsv

printf "& \\\treelstmplain "
correlate $outputs/mt-output-specific/par${metric}-constrained-treelstm.${level}level.tsv

