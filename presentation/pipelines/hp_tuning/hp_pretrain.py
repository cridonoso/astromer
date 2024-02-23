import sys
import yaml
import math
import mlflow
import mlflow.tensorflow
from functools import partial

import optuna
import os
from presentation.pipelines.hp_tuning.utils import *
from presentation.pipelines.steps import build_tf_data_loader, build_model

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from src.training.scheduler import CustomSchedule

import hydra
from omegaconf import OmegaConf

optuna.logging.set_verbosity(optuna.logging.ERROR)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
    except RuntimeError as e:
        print(e)


def train_and_evaluate(params, config, run, EXPDIR='model_dir'):
    astromer = build_model(arch='base', **params)
    # ============================================================
    if config['training']['scheduler']:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(d_model=int(params['num_heads']*params['head_dim']))
    else:
        lr = config['training']['lr']

    astromer.compile(optimizer=Adam(lr, 
                             beta_1=0.9,
                             beta_2=0.98,
                             epsilon=1e-9,
                             name='astromer_optimizer'))

    cbks = [EarlyStopping(monitor='val_loss', patience=config['training']['patience']),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1), 
                            mlflow.tensorflow.MLflowCallback( run=run)]
    tf_data = build_tf_data_loader(config['data'], 
	 							   config['training']['probed'], 
	 						       config['training']['rs'], 
                                   config['model']['maxlen'],
	 							   num_cls=None, 
	 							   batch_size=config['training']['bs'])
    astromer.fit(tf_data['train'], 
              epochs=config['training']['num_epochs'], 
              validation_data=tf_data['validation'],
              callbacks=cbks)
    print(tf_data['test_loader'])
    loss = astromer.evaluate(tf_data['test_loader'], verbose=0)
    print('LOSS:',loss)
    mlflow.tensorflow.log_model(astromer, "model")
    
    return loss 




def create_params(trial,config):
  new_dict = {}
  for key in config.keys():
    print(config[key])
    if config[key]['type'] == 'category':
        new_dict[key] = trial.suggest_categorical(key,
                                                  config[key].get('values'))
    elif config[key]['type'] == 'int':
        new_dict[key] = trial.suggest_int(key,
                                          config[key].get('low'),
                                          config[key].get('high'))
    elif config[key]['type'] == 'float':
        new_dict[key] = trial.suggest_float(key,
                                            config[key].get('low'),
                                            config[key].get('high'),
                                            log = config[key].get('log'))
    
  return  new_dict


def objective(trial, config):
    with mlflow.start_run(nested=True) as run:
        # Define hyperparameters
        params = create_params(trial, config['hp'])
        # Train model
        params.update(**config['model'])
        rmse, r2, loss = train_and_evaluate(params, config, run)


        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

    return rmse
    
def optimize_hyperparameters(config, n_trials=20):
    run_name = 'a'
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study = optuna.create_study(direction="minimize")

        study.optimize(partial(objective, config=config), n_trials=n_trials )
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val", study.best_value)
        
        mlflow.set_tags(
            tags=config['mlflow_tags']
        )

        
        print(f'Best trial score: {study.best_value}')
        print(f'Best hyperparameters: {study.best_params}')


if __name__ == "__main__":

    config = read_yaml(sys.argv[1])

    #Experiment ID
    experiment_id = get_or_create_experiment(config['mlflow_tags']['expertiment_id'])

    mlflow.set_experiment(experiment_id=experiment_id)

    optimize_hyperparameters(config)
    

    









