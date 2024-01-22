import tensorflow as tf
import seaborn as sns
# import mlflow as mf
import pandas as pd
import numpy as np
import json
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from datetime import datetime


def plot_cm(cm, ax, title='CM', fontsize=15, cbar=False, yticklabels=True, class_names=None):
    '''
    Plot Confusion Matrix
    '''
    labels = np.zeros_like(cm, dtype=np.object)
    mask = np.ones_like(cm, dtype=np.bool)
    for (row, col), value in np.ndenumerate(cm):
        if value != 0.0:
            mask[row][col] = False
        if value < 0.01:
            labels[row][col] = '< 1%'
        else:
            labels[row][col] = '{:2.1f}\\%'.format(value*100)

    ax = sns.heatmap(cm, annot = labels, fmt = '',
                     annot_kws={"size": fontsize},
                     cbar=cbar,
                     ax=ax,
                     linecolor='white',
                     linewidths=1,
                     vmin=0, vmax=1,
                     cmap='Blues',
                     mask=mask,
                     yticklabels=yticklabels)

    try:
        if yticklabels and class_names is not None:
            ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize+1)
            ax.set_xticklabels(class_names, rotation=0, fontsize=fontsize+1)
    except:
        pass
    ax.set_title(title, fontsize=fontsize+5)

    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=cm.shape[1], color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=cm.shape[0], color='k',linewidth=4)

    return ax

def get_folder_name(path, prefix=''):
    """
    Look at the current path and change the name of the experiment
    if it is repeated

    Args:
        path (string): folder path
        prefix (string): prefix to add

    Returns:
        string: unique path to save the experiment
"""

    if prefix == '':
        prefix = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])

    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if prefix not in folders:
        path = os.path.join(path, prefix)
    elif not os.path.isdir(os.path.join(path, '{}_0'.format(prefix))):
        path = os.path.join(path, '{}_0'.format(prefix))
    else:
        n = sorted([int(f.split('_')[-1]) for f in folders if '_' in f[-2:]])[-1]
        path = os.path.join(path, '{}_{}'.format(prefix, n+1))

    return path

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

def get_metrics(path_logs, metric_name='epoch_loss', full_logs=True, show_keys=False, nlog=-1):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][nlog]
    path_train = os.path.join(path_logs, train_logs)

    if full_logs:
        ea = event_accumulator.EventAccumulator(path_train, 
                                                size_guidance={'tensors': 0})
    else:
        ea = event_accumulator.EventAccumulator(path_train)
      
    ea.Reload()

    if show_keys:
        print(ea.Tags())

    try:
        metrics = pd.DataFrame([(w,s,tf.make_ndarray(t))for w,s,t in ea.Tensors(metric_name)],
                    columns=['wall_time', 'step', 'value'])
    except:
        frames = []
        for file in os.listdir(path_logs):
            if not file.endswith('.csv'): continue
            if metric_name in file:
                metrics = pd.read_csv(os.path.join(path_logs, file))
                metrics.columns = ['wall_time', 'step', 'value']
        
    return metrics

def dict_to_json(varsdic, conf_file):
    now = datetime.now()
    varsdic['exp_date'] = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)
        
        

# def mf_check_run_exists(experiment_name:str, run_name:str)->bool:
#     try:
#         experiment_meta = dict(mf.get_experiment_by_name(experiment_name))
#         experiment_id = experiment_meta["experiment_id"]
#         run = mf.MlflowClient().search_runs(experiment_ids=[str(experiment_id)],
#                                             filter_string=f"tags.`mlflow.runName` = '{run_name}'")
#         if run:
#             return True
#         else:
#             return False
#     except:
#         return False
    
# def mf_create_or_get_experiment_id(experiment_name:str):
#     # Check if the experiment exists
#     experiment = mf.get_experiment_by_name(experiment_name)

#     if experiment:
#         # If the experiment exists, return its ID
#         return experiment.experiment_id
#     else:
#         # If the experiment doesn't exist, create a new one and return its ID
#         return mf.create_experiment(experiment_name)
    
    
# def mf_set_tracking_uri(tracking_uri:str):
#     mf.set_tracking_uri(tracking_uri)
    
# def mf_set_experiment(experiment_id=None,experiment_name=None):
#     if experiment_id:
#         mf.set_experiment(experiment_id=experiment_id)
#     elif not experiment_id and experiment_name:
#         mf.set_experiment(experiment_name=experiment_name)
#     else:
#         ValueError("""Neither experiment name of experiment id is passed, please pass either of them to set the 
#                    current experiemnt""")
        

        

        
    
    
    
    
