#!/bin/bash

NUM_CHOICE=5
BINS=0

LENGTH=10
label=vam_1B_L10_C8_BIN7_FIXED7_1
#for ability in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
for ability in -1
do
    echo ======== ${ability} ========
    for s in `seq 1 5`
    do
        echo $label
        time ./vam_sample \
            --load-dir out/$label \
            --sample-no $s \
            --ability $ability \
            --num-traj 200
    done
done
