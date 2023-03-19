#!/bin/bash

NUM_CHOICE=5
LENGTH=10

for i in `seq 1 5`;
do
#    echo ======== ${i} trials \(simple\) ========
#    ./main\
#        --type simple \
#        --length ${LENGTH} \
#        --num-choices ${NUM_CHOICE} \
#        --num-traj 1000000000 \
#        --per-thread 100000 \
#        --label simple_1B_L${LENGTH}_C${NUM_CHOICE}_${i}
#
#    echo ======== ${i} trials \(unif_ability_simple\) ========
#    ./main\
#        --type unif_ability_simple \
#        --length ${LENGTH} \
#        --num-choices ${NUM_CHOICE} \
#        --num-traj 1000000000 \
#        --per-thread 100000 \
#        --label unif_ability_simple_1B_L${LENGTH}_C${NUM_CHOICE}_${i}

    for ABILITY in 0.1 0.2 0.4 0.5 0.6 0.8 0.9
    do
        echo ======== ${i} trials \(ability_simple, A${ABILITY}\) ========
        ./main\
            --type ability_simple \
            --length ${LENGTH} \
            --num-choices ${NUM_CHOICE} \
            --ability ${ABILITY} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --label ability_simple_1B_L${LENGTH}_C${NUM_CHOICE}_A${ABILITY}_${i}
    done
done
