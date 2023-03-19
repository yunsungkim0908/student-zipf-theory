#!/bin/bash

LENGTH=10
TYPE=1
BINS=2

for i in `seq 1 5`;
do
    for NUM_CHOICE in 10 20 30
    do
        echo ======== ${i} trials \(C${NUM_CHOICE}\) ========
        ./main\
            --type unif_ability_stateful \
            --length ${LENGTH} \
            --num-choices ${NUM_CHOICE} \
            --num-types ${TYPE} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --ability-bins ${BINS} \
            --label unif_ability_stateful_BINS${BINS}_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}
    done
done
