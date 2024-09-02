#!/bin/bash



nsamples=(100 1000 10000 100000 250000 500000 750000)

for str in ${nsamples[@]}; do
   echo TRAINING WITH $str 
   python -m presentation.scripts.pretrain --exp-name nsamples \
                                           --gpu 2 \
                                           --data ./data/records/snr_macho/$str/fold_0/ \
                                           --bs 2500 \
                                           --no-msk-token \
                                           --pe-base 10000 \
                                           --mask-format QK \
                                           --patience 20 \
                                           --m-alpha 1 \
                                           --lr 1e-5

done
