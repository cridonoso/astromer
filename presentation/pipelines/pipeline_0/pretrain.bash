#!/bin/bash

# nsamples=(100 1000 10000 100000 250000 500000 750000)

# for str in ${nsamples[@]}; do
#    echo TRAINING WITH $str 
#    python -m presentation.scripts.pretrain --exp-name nsamples \
#                                            --gpu 2 \
#                                            --data ./data/records/snr_macho/$str/fold_0/ \
#                                            --bs 2500 \
#                                            --no-msk-token \
#                                            --pe-base 10000 \
#                                            --mask-format QK \
#                                            --patience 20 \
#                                            --m-alpha 1 \
#                                            --lr 1e-5

# done

# alphas=(-1000000000 -1000 -100 -10 -1 0 1)

# for str in ${alphas[@]}; do
#    echo TRAINING WITH $str 
#    python -m presentation.scripts.pretrain --exp-name mask-alpha \
#                                            --gpu 1 \
#                                            --data ./data/shared/records/snr_macho/500000/fold_0 \
#                                            --bs 2500 \
#                                            --no-msk-token \
#                                            --pe-base 10000 \
#                                            --mask-format K \
#                                            --patience 20 \
#                                            --m-alpha $str \
#                                            --sampling \
#                                            --repeat 4 \
#                                            --lr 1e-5

# done


# temperatures=(0.5 1 1.5 2 2.5)

# for str in ${temperatures[@]}; do
#    echo TRAINING WITH $str 
#    python -m presentation.scripts.pretrain --exp-name temperature \
#                                            --gpu 1 \
#                                            --data ./data/shared/records/snr_macho/500000/fold_0 \
#                                            --bs 2500 \
#                                            --no-msk-token \
#                                            --pe-base 10000 \
#                                            --mask-format K \
#                                            --patience 20 \
#                                            --m-alpha -100 \
#                                            --temperature $str
#                                            --lr 1e-5

# done

probed=(0.6 0.8)

for str in ${probed[@]}; do
   echo TRAINING WITH $str PROBED
   python -m presentation.scripts.pretrain --exp-name probed \
                                           --gpu 2 \
                                           --data ./data/shared/records/snr_macho/500000/fold_0 \
                                           --bs 2500 \
                                           --no-msk-token \
                                           --pe-base 10000 \
                                           --mask-format K \
                                           --probed $str
                                           --patience 20 \
                                           --m-alpha -100 \
                                           --temperature 2.5
                                           --lr 1e-5

done

