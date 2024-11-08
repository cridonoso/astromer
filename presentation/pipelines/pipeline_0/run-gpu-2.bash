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
                 

model_paths=('./presentation/results/temperature/2024-11-05_18-57-28'
             './presentation/results/temperature/2024-11-06_02-04-44'
             './presentation/results/temperature/2024-11-06_09-20-44')

# for str in ${model_paths[@]}; do
#    echo [INFO] Testing $str
#    python -m presentation.scripts.test_model  --model $str/pretraining --gpu 2
# done

for str in ${model_paths[@]}; do
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 2
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 2
    done
done