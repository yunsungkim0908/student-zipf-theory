#!/bin/bash

NUM_CHOICE=5

for LENGTH in 10;
do
    for unif in 0.0 0.5 0.9
    do
        label=vam_sample_unif${unif}_L${LENGTH}_C${NUM_CHOICE}_${i}
        echo ======== ${label} ========
        ./vam_generate\
            --load-dist \
            --length ${LENGTH} \
            --num-choices ${NUM_CHOICE} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --uniformity ${unif} \
            --ability-bins 0 \
            --num-fixed-ability 0 \
            --label dist_only/${label}
    done
done
