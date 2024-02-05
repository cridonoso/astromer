#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_0" mlflow_tags.save_path="./presentation/results/bugstromer_0/2024-02-02_13-42-46" clf.astromer="frozen" clf.astromer_unfrozen=false
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_0" mlflow_tags.save_path="./presentation/results/bugstromer_0/2024-02-02_13-42-46" clf.astromer="unfrozen" clf.astromer_unfrozen=true

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_1" mlflow_tags.save_path="./presentation/results/bugstromer_1/2024-02-01_13-54-17" clf.astromer="frozen" clf.astromer_unfrozen=false
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_1" mlflow_tags.save_path="./presentation/results/bugstromer_1/2024-02-01_13-54-17" clf.astromer="unfrozen" clf.astromer_unfrozen=true

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_minus_0.5" mlflow_tags.save_path="./presentation/results/bugstromer_minus_0.5/2024-01-31_21-02-45" clf.astromer="frozen" clf.astromer_unfrozen=false
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_minus_0.5" mlflow_tags.save_path="./presentation/results/bugstromer_minus_0.5/2024-01-31_21-02-45" clf.astromer="unfrozen" clf.astromer_unfrozen=true

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_minus_inf" mlflow_tags.save_path="./presentation/results/bugstromer_minus_inf/2024-02-02_13-40-05" clf.astromer="frozen" clf.astromer_unfrozen=false
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="bugstromer_minus_inf" mlflow_tags.save_path="./presentation/results/bugstromer_minus_inf/2024-02-02_13-40-05" clf.astromer="unfrozen" clf.astromer_unfrozen=true
