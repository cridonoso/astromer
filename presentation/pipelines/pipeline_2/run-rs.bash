#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.4rs" mlflow_tags.save_path="./presentation/results/best_0.4rs/2024-02-12_15-29-32" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.4rs" mlflow_tags.save_path="./presentation/results/best_0.4rs/2024-02-12_15-29-32" clf.astromer="unfrozen" clf.astromer_unfrozen=true gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0rs" mlflow_tags.save_path="./presentation/results/best_0rs/2024-02-12_15-27-39" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0rs" mlflow_tags.save_path="./presentation/results/best_0rs/2024-02-12_15-27-39" clf.astromer="unfrozen" clf.astromer_unfrozen=true gpu='3'