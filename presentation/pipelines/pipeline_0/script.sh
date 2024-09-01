#!/bin/bash

model_paths=(
#              './presentation/results/maogat/2024-07-05_14-22-52'
#              './presentation/results/maogat/2024-06-02_04-10-22'
#              './presentation/results/maogat/2024-06-02_04-19-29'
#              './presentation/results/maogat/2024-06-02_04-18-39'
             './presentation/results/paper/2024-07-09_02-27-29'
#              './presentation/results/classic/2024-07-02_22-29-44'
#              './presentation/results/classic/2024-06-24_17-31-29'
#              './presentation/results/classic/2024-07-02_22-31-31'
)
             
for str in ${model_paths[@]}; do
#   echo [INFO] Starting FT $str
#   echo python -m presentation.pipelines.pipeline_0.finetune --pt-model $str --gpu 1
  echo [INFO] Starting CLF $str
  python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --gpu 1
done