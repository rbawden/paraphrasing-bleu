#!/bin/bash

hyp="$1"
ref="$2"


if [ $# -lt 2 ]; then
    echo "Wrong number of arguments specified (expecting 2)"
    echo "Usage: $0 HYPOTHESIS REF1 [REF2 REF3...REFN]"
    exit
fi


# calls sacrebleu with normal casing, default tokenisation and exp smoothing
cat $hyp | sacrebleu -sl -b --width 5 -s exp $ref 
