#!/bin/bash

NUM_CHOICE=5
LENGTH=10
BINS=2
TYPE=1

for i in `seq 1 50`
do
        echo ======== Stateful \(trial: ${i}, Uniformity: Random\) ========
        label=unif_ability_stateful_1B_UNIFrand_BINScont_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}
        echo $label
        time ./generate\
            --type unif_ability_stateful \
            --length ${LENGTH} \
            --num-choices ${NUM_CHOICE} \
            --num-types ${TYPE} \
            --num-traj 1000000000 \
            --per-thread 100000 \
            --uniformity -1 \
            --label ${label}
        
        time python plot_rank_freq.py out/${label} out/img
done

# for i in `seq 1 5`
# do
#     for UNIF in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#     do
#         echo ======== Stateful \(trial: ${i}, Uniformity: ${UNIF}\) ========
#         label=unif_ability_stateful_1B_UNIF${UNIF}_BINS${BINS}_L${LENGTH}_C${NUM_CHOICE}_T${TYPE}_${i}
#         echo $label
#         time ./main\
#             --type unif_ability_stateful \
#             --length ${LENGTH} \
#             --num-choices ${NUM_CHOICE} \
#             --num-types ${TYPE} \
#             --num-traj 1000000000 \
#             --per-thread 100000 \
#             --uniformity ${UNIF} \
#             --label ${label}
#         
#         time python plot_rank_freq.py out/${label} out/img
#     done
# done
