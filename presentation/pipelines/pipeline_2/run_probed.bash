#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="0802_inf" mlflow_tags.save_path="./presentation/results/astromer_minus_inf/2024-02-07_16-45-25" clf.astromer="frozen" clf.astromer_unfrozen=false gpu="3"

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="0102_inf" mlflow_tags.save_path="./presentation/results/astromer_minus_inf/2024-02-07_16-44-24/" clf.astromer="frozen" clf.astromer_unfrozen=false gpu="3"

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="0202_inf" mlflow_tags.save_path="./presentation/results/astromer_minus_inf/2024-02-07_16-43-39" clf.astromer="frozen" clf.astromer_unfrozen=false gpu="3"