#!/bin/bash
            
data_paths=('./data/shared/records/alcock/fold_0/alcock_20' 
            './data/shared/records/alcock/fold_1/alcock_20'
            './data/shared/records/alcock/fold_2/alcock_20'
            './data/shared/records/alcock/fold_0/alcock_100' 
            './data/shared/records/alcock/fold_1/alcock_100'
            './data/shared/records/alcock/fold_2/alcock_100'
            './data/shared/records/alcock/fold_0/alcock_500' 
            './data/shared/records/alcock/fold_1/alcock_500'
            './data/shared/records/alcock/fold_2/alcock_500'
            './data/shared/records/atlas/fold_0/atlas_20' 
            './data/shared/records/atlas/fold_1/atlas_20'
            './data/shared/records/atlas/fold_2/atlas_20'
            './data/shared/records/atlas/fold_0/atlas_100' 
            './data/shared/records/atlas/fold_1/atlas_100'
            './data/shared/records/atlas/fold_2/atlas_100'
            './data/shared/records/atlas/fold_0/atlas_500' 
            './data/shared/records/atlas/fold_1/atlas_500'
            './data/shared/records/atlas/fold_2/atlas_500')
                 
# model_paths=('./presentation/results/nsamples/2024-09-01_15-36-42'
#              './presentation/results/nsamples/2024-09-01_13-48-56'
#              './presentation/results/nsamples/2024-09-01_13-41-27'
#              './presentation/results/nsamples/2024-09-01_14-09-09'
#              './presentation/results/nsamples/2024-09-01_13-44-34')

# model_paths=('./presentation/results/new/2024-08-29_15-53-42'
#              './presentation/results/new/2024-09-01_20-04-41')

# model_paths=('./presentation/results/mask-alpha/2024-09-07_18-49-20'
#              './presentation/results/mask-alpha/2024-09-08_13-47-31'
#              './presentation/results/mask-alpha/2024-09-05_05-46-12'
#              './presentation/results/mask-alpha/2024-09-05_20-03-00'
#              './presentation/results/mask-alpha/2024-09-04_11-40-14'
#              './presentation/results/mask-alpha/2024-09-09_06-32-27'
#              './presentation/results/mask-alpha/2024-09-06_15-25-45')

# model_paths=('./presentation/results/temperature/2024-09-20_12-27-40'
#              './presentation/results/temperature/2024-09-13_15-46-31'
#              './presentation/results/temperature/2024-09-20_12-24-35'
#              './presentation/results/temperature/2024-09-16_14-45-05'
#              './presentation/results/temperature/2024-09-20_12-25-37',
#              './presentation/results/temperature/2024-09-25_10-21-22')

model_paths=('./presentation/results/probed/2024-10-02_14-23-58'
             './presentation/results/probed/2024-10-04_14-15-38'
             './presentation/results/probed/2024-10-05_13-55-24'
             './presentation/results/probed/2024-10-10_06-06-29'
             './presentation/results/probed/2024-10-11_11-59-36')

for str in ${model_paths[@]}; do
    echo [INFO] Testing $str
    python -m presentation.scripts.test_model  --model $str/pretraining --gpu 2
done

for str in ${model_paths[@]}; do
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 2
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 2
    done
done