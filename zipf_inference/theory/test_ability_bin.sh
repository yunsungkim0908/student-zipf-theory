#!/bin/bash

NUM_CHOICE=5
LENGTH=10
TYPE=1

for i in `seq 1 5`;
do
    for BINS in 2 #3 4
    do
        echo ======== ${i} trials \(stateful, ${BINS} BINS\) ========
        ./main\
            --type unif_ability_stateful \
            --length ${LENGTH} \
            --num-choices ${NUM_CHOICE} \
            --num-types ${TYPE} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --ability-bins ${BINS}\
            --label unif_BINS${BINS}_ability_stateful_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}

        # echo ======== ${i} trials \(simple, ${BINS} BINS\) ========
        # ./main\
        #     --type unif_ability_simple \
        #     --length ${LENGTH} \
        #     --num-choices ${NUM_CHOICE} \
        #     --num-types ${TYPE} \
        #     --num-traj 1000000000 \
        #     --per-thread 100000 \
        #     --ability-bins ${BINS}\
        #     --label unif_BINS${BINS}_ability_simple_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}
    done

    # echo ======== ${i} trials \(Cont.\) ========
    # ./main\
    #     --type unif_ability_stateful \
    #     --length ${LENGTH} \
    #     --num-choices ${NUM_CHOICE} \
    #     --num-types ${TYPE} \
    #     --num-traj 1000000000 \
    #     --per-thread 100000 \
    #     --label unif_ability_simple_1B_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}
done
