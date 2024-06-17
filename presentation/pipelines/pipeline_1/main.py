import mlflow.tensorflow
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import logging
import mlflow
import optuna
import toml
import sys
import os

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from functools import partial

from presentation.pipelines.steps.model_design import load_pt_model, build_classifier
from presentation.pipelines.steps.metrics import evaluate_ft, evaluate_clf
from presentation.pipelines.steps.load_data import build_loader

from presentation.pipelines.utils import get_or_create_experiment

from src.training.scheduler import CustomSchedule


optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['HYDRA_FULL_ERROR'] = '1'

tf.config.run_functions_eagerly(True)
def finetune_step(trial, run_index, config):
    # Pptuna trial numbers start from 0
    trial_number  = trial.number
    pt_folder     = config['pretrain']['weights'][run_index]
    data_path     = os.path.normpath(config['data']['paths'][trial_number])
    target_folder = os.path.normpath(config['data']['target'][trial_number])

    # Create target folder for finetuning werights
    ft_folder = os.path.join(pt_folder, 'finetuning', target_folder)
    os.makedirs(ft_folder, exist_ok=True)
    
    # This specific of target sintax (please see config_pipeline.yaml)
    data_name = target_folder.split('/')[0]
    fold_numb = target_folder.split('/')[1].split('fold_')[-1]
    setup_spc = target_folder.split('/')[2].split('_')[-1]


    with mlflow.start_run(run_name='{}_{}'.format(target_folder, trial_number), nested=True) as run:        
        tags = {
            'downstream_data': data_name,
            'spc': setup_spc,
            'fold': fold_numb
        }

        # Pretraining params
        with open(os.path.join(pt_folder, 'pretraining', 'config.toml'), 'r') as file:
            params = toml.load(file)

        # save config within the finetuning folder
        with open(os.path.join(ft_folder, "config.toml"), "w") as file:
            file.write(toml.dumps(params))

        # Save run information
        tags = {**tags, **params}
        mlflow.set_tags(tags=tags) # I think this is not necessary 
        # mlflow.log_params(params) 
        # Setup optimizer and Compile model
        if config['finetuning']['scheduler']:
            print('[INFO] Using Custom Scheduler')
            lr = CustomSchedule(d_model=int(params['num_heads']*params['head_dim']), name='scheduler')
        else:
            print('[INFO] Using Adam Optimizer')            
            lr = config['finetuning']['lr']
        
        ft_optimizer = Adam(lr, 
                            beta_1=0.9,
                            beta_2=0.98,
                            epsilon=1e-9,
                            clipnorm=1e-4,
                            name='astromer_optimizer')

        # Build Astromer model
        astromer, _ = load_pt_model(os.path.join(pt_folder, 'pretraining'), 
                                 optimizer=ft_optimizer)


        # Load downstream data 
        loaders = build_loader(data_path, 
                               params, 
                               batch_size=config['finetuning']['batch_size'],
                               clf_mode=False,
                               sampling=False,
                               normalize='zero-mean',
                               old_version=False,
                               debug=config['finetuning']['debug'],
                               return_test=True)

        # Load Callbacks
        cbks = [EarlyStopping(monitor='val_loss', patience=config['finetuning']['patience']),
                ModelCheckpoint(filepath=os.path.join(ft_folder, 'weights'),
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq='epoch',
                                verbose=1),
                mlflow.tensorflow.MLflowCallback(run=run)]

        # Finetune model
        astromer.fit(loaders['train'], 
                     batch_size=config['finetuning']['batch_size'],
                     epochs=1 if config['finetuning']['debug'] else config['finetuning']['num_epochs'], 
                     validation_data=loaders['validation'],
                     callbacks=cbks)

        # Evaluate on test
        ft_metrics = evaluate_ft(astromer, loaders['test'], params, prefix='test_')

        # Log metrics
        for key in ft_metrics.keys():
            mlflow.log_metric(key, ft_metrics[key])

        mlflow.log_artifact(os.path.join(ft_folder, "config.toml"), artifact_path="config")
        mlflow.log_artifact(local_path=ft_folder, artifact_path="model_weights")
        # mlflow.tensorflow.log_model(astromer, "ft_model", keras_model_kwargs={"save_format": "h5"})
        return ft_metrics['test_rmse']

def finetune(index, config, n_jobs=4):

    #get/create experiment
    experiment_id = get_or_create_experiment('{}_ft'.format(config['exp_id']))
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Experiment ID: {experiment_id}")

    # child_id
    child_exp_id = config['pretrain']['tag'][index]

    with mlflow.start_run(experiment_id=experiment_id, run_name=child_exp_id):
        study = optuna.create_study(direction="minimize") 

        # setup parameters of the function that will be called later
        partial_ft = partial(finetune_step, run_index=index, config=config)
        
        # optimize over the number of datasets
        n_datasets = len(config['data']['paths'])
        study.optimize(partial_ft, n_trials=n_datasets, n_jobs=n_jobs)


def classification_step(trial, run_index, config, clfarch):
    # Pptuna trial numbers start from 0
    trial_number  = trial.number
    pt_folder     = config['pretrain']['weights'][run_index]
    data_path     = os.path.normpath(config['data']['paths'][trial_number])
    target_folder = os.path.normpath(config['data']['target'][trial_number])

    # Create target folder for finetuning werights
    clf_folder = os.path.join(pt_folder, 'classification', target_folder, clfarch)
    os.makedirs(clf_folder, exist_ok=True)
    
    # This specific of target sintax (please see config_pipeline.yaml)
    data_name = target_folder.split('/')[0]
    fold_numb = target_folder.split('/')[1].split('fold_')[-1]
    setup_spc = target_folder.split('/')[2].split('_')[-1]


    with mlflow.start_run(run_name='{}_{}'.format(target_folder, trial_number), nested=True) as run:        
        tags = {
            'downstream_data': data_name,
            'spc': setup_spc,
            'fold': fold_numb
        }

        # Pretraining params
        ft_weights = os.path.join(pt_folder, 'finetuning', target_folder)
        try:
            print('[INFO] Finetuning weights detected. Loading finetuning...')
            with open(os.path.join(ft_weights, 'config.toml'), 'r') as file:
                params = toml.load(file)
        except:
            print('[INFO] No finetuning weights detected. Loading pretraining...')
            ft_weights = os.path.join(pt_folder, 'pretraining') 
            with open(os.path.join(ft_weights, 'config.toml'), 'r') as file:
                params = toml.load(file)

        tags['pt_weights'] = ft_weights

        # Save run information
        tags = {**tags, **params}
        mlflow.set_tags(tags=tags) # I think this is not necessary 
        mlflow.log_params(params) 
        mlflow.log_params({
               'clf_name': clfarch,
               'astromer_trainable': config['classification']['astromer_trainable']
        })
        # Build Astromer model
        astromer, pt_config = load_pt_model(ft_weights)

        # Load downstream data 
        loaders = build_loader(data_path, 
                               params, 
                               batch_size=config['classification']['batch_size'],
                               clf_mode=True,
                               sampling=False,
                               normalize='zero-mean',
                               old_version=False,
                               return_test=True,
                               debug=config['classification']['debug'])
        
        # Load classifier
        classifier = build_classifier(astromer, 
                                      params, 
                                      astromer_trainable=config['classification']['astromer_trainable'],
                                      num_cls = loaders['n_classes'],
                                      arch=clfarch)


        # Load Callbacks
        cbks = [EarlyStopping(monitor='val_loss', patience=config['classification']['patience']),
                ModelCheckpoint(filepath=os.path.join(clf_folder, 'weights'),
                                save_weights_only=True,
                                save_best_only=True,
                                save_freq='epoch',
                                verbose=1),
                mlflow.tensorflow.MLflowCallback(run=run)]
        
        classifier.compile(optimizer=Adam(config['classification']['lr'], name='classifier_optimizer'),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        classifier.fit(loaders['train'], 
                        epochs=1 if config['classification']['debug'] else config['classification']['num_epochs'], 
                        batch_size=config['classification']['batch_size'],
                        validation_data=loaders['validation'],
                        callbacks=cbks)

        metrics, y_true, y_pred = evaluate_clf(classifier, loaders['test'], params, prefix='test_')
                
        mlflow.log_params(params)
        for key in metrics.keys():
            mlflow.log_metric(key, metrics[key])

        table_dict = {
            "model": [config['pretrain']['tag'][run_index]],
            "dataset": [data_name],
            "fold": [fold_numb],
            "spc": [setup_spc] , 
            "clf":[clfarch],
            "clf_test_f1": [metrics['test_f1']],
            "clf_test_acc": [metrics['test_acc']],
            "clf_test_recall": [metrics['test_recall']],
            "clf_test_precision": [metrics['test_precision']]
        }

        df = pd.DataFrame.from_dict(table_dict)
        df.to_csv(os.path.join(clf_folder,"results.csv"))
        mlflow.log_table(data=df, artifact_file="results.csv")
        mlflow.tensorflow.log_model(classifier, "clf_model", keras_model_kwargs={"save_format": "h5"})
        
        with open(os.path.join(clf_folder, 'predictions.pkl'), 'wb') as handle:
            pickle.dump({'true':y_true, 'pred':y_pred}, handle)

        mlflow.log_artifact(os.path.join(clf_folder, 'predictions.pkl'), artifact_path="predictions")

        return metrics['test_f1']
   
def classify(index, config, n_jobs=1):
    experiment_id = get_or_create_experiment('{}_clf'.format(config['exp_id']))
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Experiment ID: {experiment_id}")

    # child_id
    for clfarch in config['classification']['clf_arch']:
        child_exp_id = config['pretrain']['tag'][index]+'/'+clfarch

        with mlflow.start_run(experiment_id=experiment_id, run_name=child_exp_id):
            study = optuna.create_study(direction="maximize") 

            # setup parameters of the function that will be called later
            partial_clf = partial(classification_step, run_index=index, config=config, clfarch=clfarch)
            
            # optimize over the number of datasets
            n_datasets = len(config['data']['paths'])
            study.optimize(partial_clf, n_trials=n_datasets, n_jobs=n_jobs)

    return child_exp_id

@hydra.main(version_base=None, config_path=".", config_name="config_pipeline")
def pipeline(cfg: DictConfig) -> None:
    config = cfg
    OmegaConf.set_struct(config, False)

    # Set visible GPUs for Optuna start distributing trials
    visible_gpus = ','.join(config['visible_gpu'])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    n_pretrained_models = len(config['pretrain']['tag'])
    print('[INFO] Running pipeline on {} pretrained models'.format(n_pretrained_models))
    # Our pipeline start with the finetuning step if specified
    for index in range(n_pretrained_models):
        if config['do_ft'][index]:
            finetune(index, config, n_jobs=1)

    #After finetuning has finished, we classify
    for index in range(n_pretrained_models):
        if config['do_clf'][index]:
            classify(index, config, n_jobs=1)


if __name__ == "__main__":
    '''
    
    python -m presentation.pipelines.pipeline_1.main  visible_gpu.model=[0, 1]

    '''
    pipeline()

    
    











































