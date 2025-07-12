#!/bin/bash
                 
folds=('0' '1' '2')
datasets=('atlas' 'alcock')
spcs=('20' '100' '500')
model_path='./presentation/results/diagstromer/2024-12-02_14-13-12/finetuning'

for fold_N in ${folds[@]}; do
    for dp in ${datasets[@]}; do
        for spc in ${spcs[@]}; do
            echo [INFO] Starting CLF $fold_N $spc
            python -m presentation.pipelines.referee.train \
            --pt-model $model_path/$dp/fold_$fold_N/$dp\_$spc/ --data $dp --gpu 0 --bs 512
            done
    done
done
