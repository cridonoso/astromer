#!/usr/bin/env bash

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0rs" mlflow_tags.save_path="./presentation/results/best_0rs/2024-02-12_15-27-39" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.1rs" mlflow_tags.save_path="./presentation/results/best_0.1rs/2024-02-12_15-28-34" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.4rs" mlflow_tags.save_path="./presentation/results/best_0.4rs/2024-02-12_15-29-32" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.5rs" mlflow_tags.save_path="./presentation/results/best_0.5rs/2024-02-14_21-39-44" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'

python -m presentation.pipelines.pipeline_2.main  mlflow_tags.model="best_0.8rs" mlflow_tags.save_path="./presentation/results/best_0.8rs/2024-02-12_15-33-13" clf.astromer="frozen" clf.astromer_unfrozen=false gpu='3'
