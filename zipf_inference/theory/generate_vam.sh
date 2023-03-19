#!/bin/bash

TYPE=1
UNIF=0.0
BINS=7
FIXED_BINS=7

for i in `seq 1 5`;
do
    for num_choice in 3 4 5 6 7 8 9 10 11 12 
    do
        label="vam_1B_L10_C${num_choice}_BIN${BINS}_FIXED${FIXED_BINS}_${i}"
        echo "======== ${label} ========"
        if [ -f out/$label/vals.bin ]
        then
            continue
        fi
        ./vam_test\
            --num-choices ${num_choice}\
            --length 10\
            --ability-bins ${BINS}\
            --ability-lim 0.2\
            --uniformity 0\
            --num-traj 1000000000\
            --per-thread 100000\
            --num-fixed-ability ${FIXED_BINS}\
            --label ${label}
    done

    for length in 7 8 9 10 11 12 13;
    do
        label="vam_1B_L${length}_C5_BIN${BINS}_FIXED${FIXED_BINS}_${i}"
        echo "======== ${label} ========"
        if [ -f out/$label/vals.bin ]
        then
            continue
        fi
        ./vam_test\
            --num-choices 5\
            --length ${length}\
            --ability-bins ${BINS}\
            --ability-lim 0.2\
            --uniformity 0\
            --num-traj 1000000000\
            --per-thread 100000\
            --num-fixed-ability ${FIXED_BINS}\
            --label ${label}
    done

    for num_choice in 3 4 5 6 7 8 
    do
        label="vam_1B_L10_C${num_choice}_BIN0_FIXED${FIXED_BINS}_${i}"
        echo "======== ${label} ========"
        if [ -f out/$label/vals.bin ]
        then
            continue
        fi
        ./vam_test\
            --num-choices ${num_choice}\
            --length 10\
            --ability-bins 0\
            --ability-lim 0.2\
            --uniformity 0\
            --num-traj 1000000000\
            --per-thread 100000\
            --num-fixed-ability ${FIXED_BINS}\
            --label ${label}
    done

    for length in 7 8 9 10 11 12 13;
    do
        label="vam_1B_L${length}_C5_BIN0_FIXED${FIXED_BINS}_${i}"
        echo "======== ${label} ========"
        if [ -f out/$label/vals.bin ]
        then
            continue
        fi
        ./vam_test\
            --num-choices 5\
            --length ${length}\
            --ability-bins 0\
            --ability-lim 0.2\
            --uniformity 0\
            --num-traj 1000000000\
            --per-thread 100000\
            --num-fixed-ability ${FIXED_BINS}\
            --label ${label}
    done

    for bins in 0 3 5 7 9 11
    do
        label="vam_1B_L10_C5_BIN${bins}_${i}"
        echo "======== ${label} ========"
        if [ -f out/$label/vals.bin ]
        then
            continue
        fi
        ./vam_test\
            --num-choices 5\
            --length 10\
            --ability-bins ${bins}\
            --ability-lim 0.2\
            --uniformity 0\
            --num-traj 1000000000\
            --per-thread 100000\
            --num-fixed-ability 3\
            --label ${label}
    done
done
