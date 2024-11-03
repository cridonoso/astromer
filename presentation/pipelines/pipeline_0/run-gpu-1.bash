#!/bin/bash
            
data_paths=('./data/records/alcock/fold_0/alcock_20' 
            './data/records/alcock/fold_1/alcock_20'
            './data/records/alcock/fold_2/alcock_20'
            './data/records/alcock/fold_0/alcock_100' 
            './data/records/alcock/fold_1/alcock_100'
            './data/records/alcock/fold_2/alcock_100'
            './data/records/alcock/fold_0/alcock_500' 
            './data/records/alcock/fold_1/alcock_500'
            './data/records/alcock/fold_2/alcock_500'
            './data/records/atlas/fold_0/atlas_20' 
            './data/records/atlas/fold_1/atlas_20'
            './data/records/atlas/fold_2/atlas_20'
            './data/records/atlas/fold_0/atlas_100' 
            './data/records/atlas/fold_1/atlas_100'
            './data/records/atlas/fold_2/atlas_100'
            './data/records/atlas/fold_0/atlas_500' 
            './data/records/atlas/fold_1/atlas_500'
            './data/records/atlas/fold_2/atlas_500')
                 

model_paths=('./presentation/results/nsamples/2024-11-01_16-30-33'
             './presentation/results/nsamples/2024-11-01_16-40-56')

#for str in ${model_paths[@]}; do
#    echo [INFO] Testing $str
#    python -m presentation.scripts.test_model  --model $str/pretraining --gpu 3
#done

for str in ${model_paths[@]}; do
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 1
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 1
    done
done
