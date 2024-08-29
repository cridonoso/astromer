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
            

model_paths=('./presentation/results/alpha/2024-08-27_02-00-55/')

for str in ${model_paths[@]}; do
  echo [INFO] Testing $str
  python -m presentation.scripts.test_model  --model $str/pretraining --gpu 1
  
  echo [INFO] Starting FT $str
  python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --gpu 1
  
  echo [INFO] Starting CLF $str
  python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --gpu 1
done