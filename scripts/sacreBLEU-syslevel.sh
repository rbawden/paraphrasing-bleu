#!/bin/bash

hyp="$1"
ref="$2"


if [ $# -lt 2 ]; then
    echo "Wrong number of arguments specified (expecting 2)"
    echo "Usage: $0 HYPOTHESIS REF1 [REF2 REF3...REFN]"
    exit
fi

# calls sacrebleu with casing, default tokenisation
cat $hyp | sacrebleu -b --width 5  $ref 
