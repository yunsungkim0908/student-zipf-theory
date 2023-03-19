#!/bin/bash

NUM_CHOICE=5
LENGTH=10
BINS=2
TYPE=1

for i in `seq 1 10`;
do
    echo ======== Stateful ${label} ========
    ./main\
        --type unif_ability_stateful \
        --length ${LENGTH} \
        --num-choices ${NUM_CHOICE} \
        --num-types ${TYPE} \
        --num-traj 1000000000 \
        --per-thread 100000 \
        --ability-bins ${BINS}\
        --label unif_ability_stateful_BINS${BINS}_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}

    echo ======== Simple ${label} ========
    ./main\
        --type unif_ability_simple \
        --length ${LENGTH} \
        --num-choices ${NUM_CHOICE} \
        --num-types ${TYPE} \
        --num-traj 1000000000 \
        --per-thread 100000 \
        --ability-bins ${BINS}\
        --label unif_ability_simple_BINS${BINS}_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}

done
