#!/bin/bash

hyp="$1"     # the hypothesis file
maximum="$2" # maximum number of lines to look at
ref="$3"     # the list of reference files (including additional paraphrased references)
n=$4         # the number of additional paraphrased references

if [ $# -ne 4 ]; then
    echo "Wrong number of arguments specified (expecting 2)"
    echo "Usage: $0 HYPOTHESIS MAXIMUM REF1 [REF2 REF3...REFN] NUM_EXTRA_REFS"
    exit
fi

thisdir=`dirname $0`
. $thisdir/vars

# add one (the reference)
n=$(($n + 1))

# when several refs, paste them together
ref_file=/tmp/meteor-pasting-$RANDOM
paste $ref | head -n $maximum > $ref_file.tmp
cat $ref_file.tmp | perl -pe 's/\t/\n/g' > $ref_file

hyp_file=/tmp/meteor-hyp-$RANDOM
head -n $maximum $hyp > $hyp_file


# calls meteor
java -jar $METEOR_JAR $hyp_file $ref_file -l en -r $n | grep "Segment .* score:" | perl -pe 's/^.+?\t//'

rm $ref_file.tmp
rm $ref_file
rm $hyp_file
