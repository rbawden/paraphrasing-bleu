#!/bin/bash

set -eu

lang=$1

basedir=/home/hltcoe/mpost/exp/parbleu19/paraphrases/newstest2019
DAFILE=~/exp/parbleu19/data/wmt19/wmt19-metrics-task-package/results/out/DA-newstest2019-${lang}en-sys-nohy-scores.csv
SAMPLEMIN=100
SAMPLEMAX=$(sacrebleu -t wmt19 -l $lang-en --echo src | wc -l)
SEEDMAX=10

echo "Sampling $lang from $SAMPLEMIN to $SAMPLEMAX"

[[ ! -d $lang ]] && mkdir -p $lang
cd $lang

[[ ! -d qsub.logs ]] && mkdir qsub.logs

sacrebleu -t wmt19 -l ${lang}-en --echo both | unpaste wmt19.${lang}-en.{${lang},en}
# human score
cut -d' ' -f2-3 $DAFILE | tail -n+2 > scores.human

# Create source and reference data
qsub -sync y -N sample-$lang -cwd -S /bin/bash -j y -o qsub.logs -l h_rt=24:00:00 -t 1:$SEEDMAX <<EOF
for samples in \$(seq $SAMPLEMIN 100 $SAMPLEMAX); do
    seed=\$SGE_TASK_ID
    lines=\$(cat $basedir/baseline-iter540000/${lang}en-1.en | wc -l); 
    [[ ! -d sample.\$sample/\$seed ]] && mkdir -p sample.\$samples/\$seed; 
    seq 1 \$lines | rand-sample --seed \$seed \$samples | sort -n > sample.\$samples/\$seed/lines; 

    # source and reference data
    paste wmt19.${lang}-en.{${lang},en} | filter_lines --file sample.\$samples/\$seed/lines | unpaste sample.\$samples/\$seed/wmt19.${lang}-en.{${lang},en}

    # system outputs
    for file in ~/exp/parbleu19/data/wmt19/wmt19-submitted-data/txt/system-outputs/newstest2019/${lang}-en/newstest2019.*; do
        cat \$file | filter_lines -f sample.\$samples/\$seed/lines > sample.\$samples/\$seed/\$(basename \$file);
    done

    # paraphrase outputs
    for paraphraser in baseline-iter540000 random-1235000 treelstm-plain-545000 treelstm-semhash-560000 laser-520000; do
        name=\$(echo \$paraphraser | perl -F- -ane 'pop(@F); print join("", @F);')
        paste $basedir/\$paraphraser/${lang}en-*.en | filter_lines -f sample.\$samples/\$seed/lines | unpaste sample.\$samples/\$seed/${lang}en-\$name-{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}.en;
    done
done
EOF

# get BLEU scores
qsub -N bleu-$lang -cwd -S /bin/bash -j y -o qsub.logs -l h_rt=24:00:00 -t 1:$SEEDMAX <<EOF
for sample in \$(seq $SAMPLEMIN 100 $SAMPLEMAX); do 
  seed=\$SGE_TASK_ID
  [[ -s sample.\$sample/\$seed/scores.bleu ]] && continue
  for sys in sample.\$sample/\$seed/newstest2019.*; do 
    echo -ne "\$(basename \$sys | cut -d. -f 2-3)\t"; 
    cat \$sys | sacrebleu -b \$(dirname \$sys)/wmt19.$lang-en.en
  done > sample.\$sample/\$seed/scores.bleu
done
EOF

# parbleu scores
qsub -N parbleu-$lang -cwd -S /bin/bash -j y -o qsub.logs -l h_rt=24:00:00 -t 1:$SEEDMAX <<EOF
sample=\$1
seed=\$SGE_TASK_ID
for sample in \$(seq $SAMPLEMIN 100 $SAMPLEMAX); do
    for reftype in baseline random treelstmplain treelstmsemhash laser; do
        for numrefs in 1 5 10 20; do
            [[ -s sample.\$sample/\$seed/scores.parbleu.\$reftype.\$numrefs ]] && continue
            refstr=""
            for refno in \$(seq 1 \$numrefs); do
                refstr+="sample.\$sample/\$seed/${lang}en-\$reftype-\$refno.en "
            done
            for sys in sample.\$sample/\$seed/newstest2019.*; do 
                echo -ne "\$(basename \$sys | cut -d. -f 2-3)\t"; 
                cat \$sys | sacrebleu -b \$(dirname \$sys)/wmt19.$lang-en.en \$refstr
            done > sample.\$sample/\$seed/scores.parbleu.\$reftype.\$numrefs
        done
    done
done
EOF
