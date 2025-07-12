#!/bin/bash
                 
folds=('0' '1' '2')
datasets=('atlas' 'alcock')
spcs=('20' '100' '500')
model_path='./presentation/results/diagstromer/2024-12-02_14-13-12/finetuning'
clf_models=('max' 'avg' 'skip' 'att_avg' 'att_cls')
for fold_N in ${folds[@]}; do
    for dp in ${datasets[@]}; do
        for spc in ${spcs[@]}; do
            for clfmodel in ${clf_models[@]}; do
                echo [INFO] Starting CLF $fold_N $spc
                python -m presentation.pipelines.referee.train \
                --pt-path $model_path/$dp/fold_$fold_N/$dp\_$spc/ --data ./data/records/$dp/$dp\_$spc/fold_$fold_N --gpu 0 --bs 512 \
                --exp-name clf_$dp\_$fold_N\_$spc --clf-arch $clfmodel
                done
            done
    done
done
