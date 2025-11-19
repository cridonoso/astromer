#!/bin/bash
                 
folds=('1')
datasets=('alcock')
spcs=('500')
model_path='./presentation/results/diagstromer/2024-12-02_14-13-12/finetuning'
# 'base_avgpool' 'base_gru' 'max' 'avg' 'skip' 'att_avg' 'att_cls'
clf_models=('avg' 'skip')
for fold_N in ${folds[@]}; do
    for dp in ${datasets[@]}; do
        for spc in ${spcs[@]}; do
            for clfmodel in ${clf_models[@]}; do
                echo [INFO] Starting CLF $fold_N $spc $clfmodel
                python -m presentation.pipelines.referee.train \
                --pt-path $model_path/$dp/fold_$fold_N/$dp\_$spc/ \
                --data ./data/records/$dp/$spc/fold_$fold_N \
                --gpu 1 \
                --bs 512 \
                --exp-name clf_$dp\_$fold_N\_$spc --clf-arch $clfmodel
                done
            done
    done
done
