#!/bin/bash

NUM_CHOICE=5
LENGTH=10
BINS=2
TYPE=1

i=1
for label in out/vam_sample_L10_C5_A_*
do
    for s in `seq 1 5`
    do
        echo $label
        time ./sample \
            --type unif_ability_stateful \
            --load-dir $label \
            --sample-no $s
    done
done
