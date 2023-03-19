#!/bin/bash

TYPE=1
UNIF=0.0

for i in `seq 1 5`;
do
    for num_choice in 2 4 8 16 32 48 64
    do
        for ability in 0.5 0.8 0.2;
        do
            echo ======== test_1B_L10_C${num_choice}_A${ability}_${i} ========
            if [ -f out/test_1B_L10_C${num_choice}_A${ability}_${i}/vals.bin ]
            then
                continue
            fi
            ./generate\
                --type ability_stateful \
                --num-choices ${num_choice} \
                --length 10 \
                --num-type 1 \
                --ability ${ability} \
                --num-traj 1000000000 \
                --per-thread 100000 \
                --label test_1B_L10_C${num_choice}_A${ability}_${i}
        done
    done
    for num_choice in 16 32 48 64
    do
        for bins in 0 2 4;
        do
            echo ======== test_1B_BINS${bins}_L10_C${num_choice}_${i} ========
            if [ -f out/test_1B_BINS${bins}_L10_C${num_choice}_${i}/vals.bin ]
            then
                continue
            fi
            ./generate\
                --type unif_ability_stateful \
                --num-choices ${num_choice} \
                --length 10 \
                --num-types 1 \
                --ability-bins ${bins}\
                --uniformity ${UNIF} \
                --num-traj 1000000000 \
                --per-thread 100000 \
                --label test_1B_BINS${bins}_L10_C${num_choice}_${i}
        done
    done
done
