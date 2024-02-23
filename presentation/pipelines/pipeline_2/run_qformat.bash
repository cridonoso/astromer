#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="qformat_0" mlflow_tags.save_path="./presentation/results/qformat_0/2024-02-08_13-16-32" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="qformat_1" mlflow_tags.save_path="./presentation/results/qformat_1/2024-02-08_13-17-52" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="qformat_m0.5" mlflow_tags.save_path="./presentation/results/qformat_minus_0.5/2024-02-08_03-46-28" clf.astromer="frozen" clf.astromer_unfrozen=false gpu="3"

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="qformat_minf" mlflow_tags.save_path="./presentation/results/qformat_minus_inf/2024-02-07_18-39-30" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
