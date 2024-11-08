#!/bin/bash
            
data_paths=('./data/records/alcock/20/fold_0' 
            './data/records/alcock/20/fold_1'
            './data/records/alcock/20/fold_2'
            './data/records/alcock/20/fold_3'
            './data/records/alcock/20/fold_4'
            './data/records/alcock/100/fold_0' 
            './data/records/alcock/100/fold_1'
            './data/records/alcock/100/fold_2'
            './data/records/alcock/100/fold_3'
            './data/records/alcock/100/fold_4'
            './data/records/alcock/500/fold_0' 
            './data/records/alcock/500/fold_1'
            './data/records/alcock/500/fold_2'
            './data/records/alcock/500/fold_3'
            './data/records/alcock/500/fold_4')
                 

model_paths=('./presentation/results/temperature/2024-11-05_14-28-14'
             './presentation/results/temperature/2024-11-05_14-28-35'
             './presentation/results/temperature/2024-11-05_17-13-54')

for str in ${model_paths[@]}; do
   echo [INFO] Testing $str
   python -m presentation.scripts.test_model  --model $str/pretraining --gpu 1
done

for str in ${model_paths[@]}; do
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 1
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 1
    done
done