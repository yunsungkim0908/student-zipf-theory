#!/bin/bash

TYPE=1
UNIF=0.0

for i in `seq 1 5`;
do
    for len in 6 8 10 12 14
    do
        for ability in 0.5 0.8 0.2;
        do
            echo ======== test_1B_L${len}_C5_A${ability}_${i} ========
            if [ -f out/test_1B_L${len}_C5_A${ability}_${i}/vals.bin ]
            then
                continue
            fi
            ./generate\
                --type ability_stateful \
                --num-choices 5 \
                --length ${len} \
                --num-type 1 \
                --ability ${ability} \
                --num-traj 1000000000 \
                --per-thread 100000 \
                --label test_1B_L${len}_C5_A${ability}_${i}
        done
        for bins in 0 2;
        do
            echo ======== test_1B_BINS${bins}_L${len}_C5_${i} ========
            if [ -f out/test_1B_BINS${bins}_L${len}_C5_${i}/vals.bin ]
            then
                continue
            fi
            ./generate\
                --type unif_ability_stateful \
                --num-choices 5 \
                --length ${len} \
                --num-types 1 \
                --ability-bins ${bins}\
                --uniformity ${UNIF} \
                --num-traj 1000000000 \
                --per-thread 100000 \
                --label test_1B_BINS${bins}_L${len}_C5_${i}
        done
    done
done
