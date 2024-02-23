#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="nsp" mlflow_tags.save_path="./presentation/results/nsp/2024-02-15_14-32-32" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="nsp" mlflow_tags.save_path="./presentation/results/nsp/2024-02-15_14-32-32" clf.astromer="unfrozen" clf.astromer_unfrozen=true gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="skip" mlflow_tags.save_path="./presentation/results/skip/2024-02-15_14-32-52/" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="skip" mlflow_tags.save_path="./presentation/results/skip/2024-02-15_14-32-52/" clf.astromer="unfrozen" clf.astromer_unfrozen=true gpu='3'