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
                 

model_paths=('./presentation/results/m_alpha/2024-11-05_14-18-53'
             './presentation/results/m_alpha/2024-11-06_01-49-31'
             './presentation/results/m_alpha/2024-11-06_06-52-07')

# for str in ${model_paths[@]}; do
#    echo [INFO] Testing $str
#    python -m presentation.scripts.test_model  --model $str/pretraining --gpu 0
# done

for str in ${model_paths[@]}; do
    for dp in ${data_paths[@]}; do
        echo [INFO] Starting FT $str
        python -m presentation.pipelines.pipeline_0.finetune --pt-model $str/pretraining --data $dp --gpu 0
        
        echo [INFO] Starting CLF $str
        python -m presentation.pipelines.pipeline_0.classify --pt-model $str/pretraining --data $dp --gpu 0
    done
done
