#!/bin/bash

hyp="$1" # the hypothesis file
ref="$2" # the list of reference files (including additional paraphrased references)
n=$3     # the number of additional paraphrased references 

if [ $# -ne 3 ]; then
    echo "Wrong number of arguments specified (expecting 2)"
    echo "Usage: $0 HYPOTHESIS REF1 [REF2 REF3...REFN] NUM_EXTRA_REFS"
    exit
fi

thisdir=`dirname $0`
. $thisdir/vars

# add one (the reference)
n=$(($n + 1))

# when several refs, paste them together
ref_file=/tmp/meteor-pasting-$RANDOM
paste -d '\n' $ref > $ref_file

# calls meteor
java -jar $METEOR_JAR $hyp $ref_file -l en -r $n | grep "Segment .* score:" | perl -pe 's/^.+?\t//'

rm $ref_file
