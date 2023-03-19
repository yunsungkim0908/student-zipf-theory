#!/bin/bash

for i in `seq 1 10`;
do
    for NUM_CHOICE in 3 4 5 6 7
    do
        label="monkey_1B_C${NUM_CHOICE}_${i}"

        echo ======== ${label} ========
        ./main\
            --type monkey \
            --num-choices ${NUM_CHOICE} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --label ${label}
    done
done
