import sys
import os
import toml
import optuna
import mlflow
import mlflow.tensorflow
from functools import partial
import pandas as pd
from presentation.pipelines.hp_tuning.utils import *
from presentation.pipelines.steps import build_tf_data_loader, build_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from src.training.scheduler import CustomSchedule
from presentation.experiments.utils import * 
import tensorflow as tf
from src.models.astromer_0 import  build_input
import logging
import numpy as np

optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import hydra
from omegaconf import DictConfig, OmegaConf

#mlflow.tensorflow.autolog( )

os.environ['HYDRA_FULL_ERROR'] = '1'


def clf_train_evaluate(trial, exp_id, spc_id, config):
    # optuna trial numbers start from 0
    trial_number = trial.number
    folds = config['experiment']['folds'][trial_number]
    
    # save dir child
    EXPDIR = os.path.join(config['mlflow_tags']['save_path'], 
                            'classification', config['clf']['astromer'] ,exp_id,
                             '{}_{}'.format(exp_id,spc_id), folds)
    os.makedirs(EXPDIR, exist_ok=True)
    
    # fold training
    with mlflow.start_run(run_name=folds, nested=True) as run:

        config['mlflow_tags']['ft_folder'] = EXPDIR
        config['mlflow_tags']['data_path'] = os.path.join(config['data'], exp_id, folds, '{}_{}'.format(exp_id,spc_id))
        mlflow.set_tags(tags=config['mlflow_tags'])

        # load pretrain astromer cf
        with open(os.path.join(config['mlflow_tags']['save_path'],'pretraining','config.toml'), 'r') as file:
            params = toml.load(file)


        astromer = build_model(params)
        astromer.load_weights(os.path.join(config['mlflow_tags']['save_path'], 
                            'finetuning', exp_id,'{}_{}'.format(exp_id,spc_id), folds, 'weights')).expect_partial()
        
        lr = config['training']['lr']
        num_cls = pd.read_csv(os.path.join(config['mlflow_tags']['data_path'], 'objects.csv')).shape[0]
        tf_data = build_tf_data_loader(config['mlflow_tags']['data_path'],
                                       params, 
	 							       batch_size=config['training']['bs'],
                                       clf_mode=True)
        
        cbks = [EarlyStopping(monitor='val_loss', patience=config['training']['patience']),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1),
                             mlflow.tensorflow.MLflowCallback( run=run)]
        
        inp_placeholder = build_input(config['training']['maxlen'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = config['clf']['astromer_unfrozen']
        embedding = encoder(inp_placeholder)
        mask = 1.- inp_placeholder['mask_in']
        
        clf = Model(inputs=inp_placeholder, outputs= get_mlp_avg(embedding, mask, num_cls), name='mlp_att')
        clf.compile(optimizer=Adam(lr, 
                             beta_1=0.9,
                             beta_2=0.98,
                             epsilon=1e-9,
                             name='astromer_optimizer'),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    
        clf.fit(tf_data['train'], 
                epochs=config['training']['num_epochs'], 
                validation_data=tf_data['validation'],
                callbacks=cbks)
        
        clf.load_weights(os.path.join(EXPDIR, 'weights')).expect_partial()
        y_pred = clf.predict(tf_data['test_loader'])
        y_true = tf.concat([y for _, y in tf_data['test_loader']], 0)
        pred_labels = tf.argmax(y_pred, 1)
        true_labels = tf.argmax(y_true, 1)
        p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                    pred_labels,
                                                    average='macro',
                                                    zero_division=0.)
        test_acc = accuracy_score(true_labels, pred_labels)

        #loss, acc = clf.evaluate(tf_data['test_loader'], verbose=0)
        mlflow.log_params(params)
        
        mlflow.log_metric("clf_test_f1", f)
        mlflow.log_metric("clf_test_acc", test_acc)
        mlflow.log_metric("clf_test_recall", r)
        mlflow.log_metric("clf_test_precision", p)
        
        table_dict = {
            "model": [config['mlflow_tags']['model']],
            "dataset": [exp_id],
            "spc": [spc_id] , 
            "astromer":[config['clf']['astromer']],
            "clf":[config['clf']['arch']],
            "fold": [trial_number],
            "clf_test_f1": [f],
            "clf_test_acc": [test_acc],
            "clf_test_recall": [r],
            "clf_test_precision": [p]
        }
        df = pd.DataFrame.from_dict(table_dict)
        df.to_csv(os.path.join(EXPDIR,"results.csv"))
        mlflow.log_table(data=df, artifact_file="results.csv")
        #mlflow.tensorflow.log_model(clf, "clf_model",keras_model_kwargs={"save_format": "h5"})

        
        with open(os.path.join(EXPDIR, 'predictions.pkl'), 'wb') as handle:
            pickle.dump({'true':y_true, 'pred':y_pred}, handle)

        mlflow.log_artifact(os.path.join(EXPDIR, 'predictions.pkl'), artifact_path="predictions")


        return f



def ft_train_evaluate(trial, exp_id, spc_id, config):
    # optuna trial numbers start from 0
    trial_number = trial.number
    folds = config['experiment']['folds'][trial_number]
    
    # save dir child
    EXPDIR = os.path.join(config['mlflow_tags']['save_path'], 
                            'finetuning', exp_id,
                             '{}_{}'.format(exp_id,spc_id), folds)
    os.makedirs(EXPDIR, exist_ok=True)
    
    # fold training
    with mlflow.start_run(run_name=folds, nested=True) as run:

        config['mlflow_tags']['ft_folder'] = EXPDIR
        config['mlflow_tags']['data_path'] = os.path.join(config['data'], exp_id, folds, '{}_{}'.format(exp_id,spc_id))
        mlflow.set_tags(tags=config['mlflow_tags'])

        # load pretrain astromer cf
        with open(os.path.join(config['mlflow_tags']['save_path'],'pretraining','config.toml'), 'r') as file:
            params = toml.load(file)

        astromer = build_model(params)


        if config['training']['scheduler']:
            print('[INFO] Using Custom Scheduler')
            lr = CustomSchedule(d_model=int(params['num_heads']*params['head_dim']))
        else:
            lr = config['training']['lr']

        tf_data = build_tf_data_loader(config['mlflow_tags']['data_path'],
                                       params, 
	 							       batch_size=config['training']['bs'],
                                       clf_mode=False)
        
        astromer.compile(optimizer=Adam(lr, 
                             beta_1=0.9,
                             beta_2=0.98,
                             epsilon=1e-9,
                             name='astromer_optimizer'))
        astromer.load_weights(os.path.join(config['mlflow_tags']['save_path'], 'pretraining', 'weights')).expect_partial()

        cbks = [EarlyStopping(monitor='val_loss', patience=config['training']['patience']),
                ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1),
                mlflow.tensorflow.MLflowCallback( run=run)]
        
        astromer.fit(tf_data['train'], 
              epochs=config['training']['num_epochs'], 
              validation_data=tf_data['validation'],
              callbacks=cbks)
        rmse, r2, loss = astromer.evaluate(tf_data['test_loader'])

        # save config as parametros
        mlflow.log_params(params) 
        # save config as config.toml
        with open(os.path.join(EXPDIR, "config.toml"), "w") as file:
            file.write(toml.dumps(params))

        mlflow.log_artifact(os.path.join(EXPDIR, "config.toml"), artifact_path="config")
        mlflow.log_artifact(local_path=EXPDIR, artifact_path="model_weights")
       # mlflow.tensorflow.log_model(astromer, "ft_model", keras_model_kwargs={"save_format": "h5"})
   
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        return rmse


def finetune(exp_id, config):

    #get/create experiment
    experiment_id = get_or_create_experiment('ft_{}'.format(exp_id))
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Experiment ID: {experiment_id}")

    for spc in config['experiment']['spc']:
        #parent id
        parent_id = 'ft_{}_{}_{}'.format(exp_id,spc,config['mlflow_tags']['model'])
        with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=parent_id):
            
            #add tags to the parent node
            config['mlflow_tags']['dataset'] = exp_id
            config['mlflow_tags'].update(config['training'])
            mlflow.set_tags(tags=config['mlflow_tags'])
            
            #train n evaluate - childs
            study = optuna.create_study(direction="minimize") 
            study.optimize(partial(ft_train_evaluate, exp_id=exp_id, config=config, spc_id=spc), len(config['experiment']['folds']))
            
            #parent info
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_fold_rmse", study.best_value)
   
def classify(exp_id, config):

    #get/create experiment
    experiment_id = get_or_create_experiment('clf_{}'.format(exp_id))
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Experiment ID: {experiment_id}")

    for spc in config['experiment']['spc']:
        #parent id
        parent_id = 'clf_{}_{}_{}_{}_{}'.format(exp_id,spc, config['clf']['astromer'], config['mlflow_tags']['model'], config['clf']['arch'])
        with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=parent_id):
            
            #add tags to the parent node
            config['mlflow_tags']['dataset'] = exp_id
            config['mlflow_tags'].update(config['clf'])
            mlflow.set_tags(tags=config['mlflow_tags'])
            
            #train n evaluate - childs
            study = optuna.create_study(direction="maximize") 
            study.optimize(partial(clf_train_evaluate, exp_id=exp_id, config=config, spc_id=spc), len(config['experiment']['folds']))
            
            trials =  [trial.value for trial in study.trials if trial.value is not None]
            
            #parent info
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_fold_f1", study.best_value)
            mlflow.log_metric("mean_f1", np.mean(trials))
            mlflow.log_metric("std_f1",  np.std(trials))


@hydra.main(version_base=None, config_path=".", config_name="config_pipeline")
def pipeline(cfg: DictConfig) -> None:
    config = cfg
    OmegaConf.set_struct(config, False)
    

    for exp_id, ft in zip(config['expertiment_ids'], config['do_ft']):
        if ft:
            finetune(exp_id, config)
    
    for exp_id, clf in zip(config['expertiment_ids'], config['do_clf']):
        if clf:
            classify(exp_id, config)


if __name__ == "__main__":

    pipeline()

    
    











































