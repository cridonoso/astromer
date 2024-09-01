#!/bin/bash

# model_paths=('./presentation/results/macho_1e3/2024-08-06_10-22-38/'
#              './presentation/results/macho_1e3/2024-08-05_00-59-21/')
# model_paths=('./presentation/results/macho_100/2024-07-18_18-21-35'
#              './presentation/results/macho_1000/2024-07-18_20-15-41'
#              './presentation/results/macho_10000/2024-07-18_20-16-07'
#              './presentation/results/macho_100000/2024-07-19_10-03-35'
#              './presentation/results/macho_500000/2024-07-23_14-47-04'
#              './presentation/results/macho_500000/2024-07-30_09-04-21'
#              './presentation/results/macho_500000/2024-07-30_09-04-45'
#              './presentation/results/macho_750000/2024-07-23_14-45-39'
#              './presentation/results/macho_1000000/2024-07-23_14-46-24')
            
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
                 
model_paths=('./presentation/results/dist-keras/2024-08-30_16-19-05/')

for str in ${model_paths[@]}; do
    echo [INFO] Testing $str
    # python -m presentation.scripts.test_model  --model $str/pretraining --gpu 0
    
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 0
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 0
    done
done