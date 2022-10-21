import tensorflow as tf
import pandas as pd
import shutil
import tomli
import os

from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam

from time import gmtime, strftime, time

from src.models.zero import get_ASTROMER
from src.data import pretraining_pipeline
from src.training import CustomSchedule

print('AQUIIII::::: ',os.getcwd())

def create_or_load_history(path='./results/history.csv', id=None):
    """
    Create a pandas DataFrame to store pipeline history.
    Pipeline histories contain information about when and who execute an
    experiment as well as the elapsed time.

    Args:
        path (string): Path to save dataframe as .csv

    Returns:
        Pandas DataFrame: Table containing information about the pipeline runtime
        id (integer): Identifer asociated to the last register.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        id = df.id.iloc[-1] if id is None else id
    else:
        df = pd.DataFrame(columns=['id','date', 'owner',
                                   'config_file', 'status',
                                   'elapsed'])
        id = 0 if id is None else id

    return df, id

def report_history(df, path, **kwargs):
    """
    Writes the current status of the pipeline.
    Notice **kwargs can be any column in the original DataFrame,
    i.e., columns defined at create_or_load_history
    e.g., status='this will overwrites the partial dictionary'

    Args:
        df (string): DataFrame containing the history
                     (hint: depends on create_or_load_history)
        path (string): Path to save 'df' the history .csv file.

    Returns:
        Pandas DataFrame: An updated history DataFrame.
    """
    partial = {'id':0,
               'date': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
               'owner': os.getenv("HOST"),
               'config_file': '',
               'status':'init',
               'elapsed':0.}

    for key, arg in kwargs.items():
        partial[key] = arg

    current = pd.DataFrame(partial, index=[0])
    df = pd.concat([df, current])
    df.to_csv(path, index=False)
    return df

def create_target_directory(config_file, path):
    """
    Create a folder to store weights, metrics and configuration files.

    Args:
        path (string): directory path
        config_file (string, default=None): Path to the config. file
    """
    # Create folder to store weights and metrics
    os.makedirs(path, exist_ok=True)
    if config_file is not None:
        dst = os.path.join(path, 'config.toml')
        if not os.path.exists(dst):
            shutil.copy(config_file, dst)

def load_pt_data(config,
                 subsets=['train', 'val', 'test'],
                 step='pretraining'):
    """
    Given a config file, this method loads data following the
    the pre-training format (i.e., Masking)
    Only support records files.

    Args:
        config (dictonary): Configuration of the pipeline experiment
        subsets (list): subsets to load. Notice the names of the subsets match
                        the name of the records files
        step: finetuning or pretraining
    Returns:
        dictonary: A dictonary containing tf.Dataset for each subset.
    """

    data = dict()
    print('[INFO] Loading data from {}'.format(config[step]['data']['path']))
    for subset in subsets:
        repeat = config[step]['data']['repeat'] if subset == 'train' else None
        data[subset] = pretraining_pipeline(os.path.join(config[step]['data']['path'], subset),
                                            config[step]['data']['batch_size'],
                                            config['astromer']['window_size'],
                                            config['masking']['mask_frac'],
                                            config['masking']['rnd_frac'],
                                            config['masking']['same_frac'],
                                            config[step]['data']['sampling'],
                                            config[step]['data']['shuffle_{}'.format(subset)],
                                            repeat=repeat,
                                            num_cls=None,
                                            normalize=config[step]['data']['normalize'],
                                            cache=config[step]['data']['cache_{}'.format(subset)])
    return data

def compile_astromer(config, model, step='pretraining'):
    """
    Compile an astromer instance (model) by setting up the optimizer and
    loading weights if defined.

    Args:
        config (dictonary): Configuration of the pipeline experiment

    Returns:
        astromer model compiled
    """
    if step == 'classification' or not config[step]['scheduler']:
        print('[INFO] Using learning rate: {}'.format(config[step]['lr']))
        lr = config[step]['lr']
    else:
        print('[INFO] Using custom scheduler')
        model_dim = config['astromer']['head_dim']*config['astromer']['heads']
        lr = CustomSchedule(model_dim)


    optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer)
    if step == 'finetuning':
        print('[INFO] Pretrained weights: {}'.format(config[step]['weights']))
        model.load_weights(os.path.join(config[step]['weights'], 'weights'))
    return model

def get_callbacks(config, step='pretraining', monitor='val_loss', extra=''):
    """
    Create useful callbacks (the same for pretraining, finetuning and classification)
    Notice the monitor should match the name of the optimization metric.

    Args:
        config (dictonary): description
        step (string): pretraining, finetuning, or classification
        extra: extra folder to separate experiment with the same exp_path
    Returns:
        type: description

    Raises:
        Exception: description

    """

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config[step]['exp_path'], extra, 'weights'),
            save_weights_only=True,
            monitor=monitor,
            save_best_only=True),
        EarlyStopping(monitor=monitor,
            patience = config[step]['patience'],
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(config[step]['exp_path'], extra, 'logs'),
            histogram_freq=1,
            write_graph=True)]
    return callbacks

def save_metrics(metrics, path=None):
    """
    Save a metrics dictonary as .csv table

    Args:
        metrics (dictonary): A dictonary contining the key (metric name)
                             and the value (metric's score)
        path (string): Path to write the .csv file
    """
    current = pd.DataFrame(metrics, index=[0])

    # Initialize history or use a previous one
    if os.path.exists(path):
        print('[INFO] File {} already exists. Concat mode.'.format(path))
        metrics_df = pd.read_csv(path)
    else:
        metrics_df = pd.DataFrame(columns=list(metrics.keys()))

    metrics_df = pd.concat([metrics_df, current])
    metrics_df.to_csv(path, index=False)

def load_metrics(path=None):
    """
    Load pandas dataframe with the metrics of the pipeline execution.

    Args:
        path (string): .csv filepath. By default this file is named
                       'metrics.csv' and is located in the 'exp_path' folder,
                       defined at the config_file.

    Returns:
        type: Pandas dataframe with the (collection of) metrics

    """
    metrics_df = pd.read_csv(path)
    return metrics_df

def load_clf_data(config, subsets=['train', 'val', 'test']):
    """
    Load data for classification i.e., with class labels
    Only support records files.
    Args:
        config (dictonary): Configuration of the pipeline experiment
        subsets (list): subsets to load. Notice the names of the subsets match
                        the name of the records files
    Returns:
        dictonary: A dictonary containing tf.Dataset for each subset.
    """
    num_cls = pd.read_csv(
                os.path.join(config['classification']['data']['path'],
                            'objects.csv')).shape[0]
    data = dict()
    for subset in subsets:
        repeat = config['classification']['data']['repeat'] if subset == 'train' else None
        data[subset] = pretraining_pipeline(
                os.path.join(config['classification']['data']['path'], subset),
                config['classification']['data']['batch_size'],
                config['astromer']['window_size'],
                0.,
                0.,
                0.,
                False,
                config['classification']['data']['shuffle_{}'.format(subset)],
                repeat=repeat,
                num_cls=num_cls,
                normalize=config['classification']['data']['normalize'],
                cache=config['classification']['data']['cache_{}'.format(subset)])
    return data
