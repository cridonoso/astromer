'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import pandas as pd
import tomli
import os

from src.models.classifiers.paper_0 import get_classifier_by_name
from src.data import pretraining_pipeline
from presentation.pipeline.base import *
from time import gmtime, strftime, time

from tensorflow.keras.callbacks  import EarlyStopping, TensorBoard
from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def train(config_file, history_path='', step='pretraining', pipeline_id=None):
    '''
    pretraining/finetuning pipeline
    '''
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    create_target_directory(config_file=config_file,
                            path=config[step]['exp_path'])

    df, id = create_or_load_history(path=history_path, id=pipeline_id)
    df = report_history(df, history_path,
                        id=id,
                        status='init_{}'.format(step),
                        config_file=config_file,
                        elapsed=time() - start)

    # Creating ASTROMER
    d_model = config['astromer']['head_dim']*config['astromer']['heads']
    astromer =  get_ASTROMER(num_layers=config['astromer']['layers'],
                             d_model=d_model,
                             num_heads=config['astromer']['heads'],
                             dff=config['astromer']['dff'],
                             base=config['positional']['base'],
                             dropout=config['astromer']['dropout'],
                             maxlen=config['astromer']['window_size'],
                             no_train=False)
    astromer = compile_astromer(config, astromer, step=step)
    df = report_history(df, history_path,
                        id=id,
                        status='astromer_created',
                        config_file=config_file,
                        elapsed=time() - start)
    # Get callbacks
    cbks = get_callbacks(config, monitor='val_loss')

    # Loading data
    data = load_pt_data(config, subsets=['train', 'val', 'test'], step=step)
    df = report_history(df, history_path,
                        id=id,
                        status='data_loaded',
                        config_file=config_file,
                        elapsed=time() - start)

    # Train ASTROMER
    _ = astromer.fit(data['train'],
                  epochs=config[step]['epochs'],
                  validation_data=data['val'],
                  callbacks=cbks)
    df = report_history(df, history_path,
                        id=id,
                        status='{}_done'.format(step),
                        config_file=config_file,
                        elapsed=time()-start)

    # Getting metrics
    loss, r2 = astromer.evaluate(data['test'])
    metrics = {'rmse':loss, 'r_square':r2}
    save_metrics(metrics,
                 path=os.path.join(config[step]['exp_path'],
                                   'metrics.csv'))
    df = report_history(df, history_path,
                        id=id,
                        status='test_{}_done'.format(step),
                        config_file=config_file,
                        elapsed=time() - start)
    return id

def classify(config_file, history_path, pipeline_id=None):
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    create_target_directory(config_file=config_file,
                            path=config['classification']['exp_path'])

    df, id = create_or_load_history(path=history_path, id=pipeline_id)
    df = report_history(df, history_path, id=id,
                        status='clf_init',
                        config_file=config_file,
                        elapsed=time() - start)

    # Load data for classification
    data = load_clf_data(config)
    df = report_history(df, history_path, id=id,
                        status='data_loaded',
                        config_file=config_file,
                        elapsed=time() - start)

    # Load pre-trained model
    d_model = config['astromer']['head_dim']*config['astromer']['heads']
    astromer =  get_ASTROMER(num_layers=config['astromer']['layers'],
                             d_model=d_model,
                             num_heads=config['astromer']['heads'],
                             dff=config['astromer']['dff'],
                             base=config['positional']['base'],
                             dropout=config['astromer']['dropout'],
                             maxlen=config['astromer']['window_size'],
                             no_train=False)
    astromer = compile_astromer(config, astromer, step='classification')
    df = report_history(df, history_path, id=id,
                        status='astromer_created',
                        config_file=config_file,
                        elapsed=time() - start)

    for clf_name in ['mlp_att', 'lstm_att', 'lstm']:
        print('[INFO] Training {}'.format(clf_name))
        df = report_history(df, history_path, id=id,
                            status='training_{}'.format(clf_name),
                            config_file=config_file,
                            elapsed=time() - start)
        clf_model = get_classifier_by_name(clf_name,
                    config,
                    astromer=astromer,
                    train_astromer=config['classification']['train_astromer'])

        # Compile and train
        optimizer = Adam(learning_rate=config['classification']['lr'])
        os.makedirs(config['classification']['exp_path'], exist_ok=True)

        clf_model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics='accuracy')

        cbks = get_callbacks(config, step='classification',
                             monitor='val_loss', extra=clf_name)

        _ = clf_model.fit(data['train'],
                          epochs=config['classification']['epochs'],
                          callbacks=cbks,
                          validation_data=data['val'])

        clf_model.save(os.path.join(config['classification']['exp_path'],
                       clf_name, 'model'))

        # Evaluate
        df = report_history(df, history_path, id=id,
                            status='testing_{}'.format(clf_name),
                            config_file=config_file,
                            elapsed=time() - start)

        loss, acc = clf_model.evaluate(data['test'])
        metrics = {'loss':loss, 'acc':acc}
        # Save metrics
        save_metrics(metrics,
                     path=os.path.join(config['classification']['exp_path'],
                                       'metrics.csv'))

    df = report_history(df, history_path, id=id,
                        status='clf_done'.format(clf_model),
                        config_file=config_file,
                        elapsed=time() - start)
    return id

if __name__ == '__main__':

    config_file = './pipeline/config/template.toml'

    # id = train(config_file,
    #            history_path='./results/history.csv',
    #            step='pretraining')

    id = train(config_file,
               history_path='./results/history.csv',
               step='finetuning')

    id= classify(config_file,
                 history_path='./results/history.csv',
                 pipeline_id=id)
