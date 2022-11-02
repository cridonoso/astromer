'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import pandas as pd
import tomli
import os, sys

from src.models.classifiers.paper_0 import get_classifier_by_name
from src.data import pretraining_pipeline
from src.pipeline.base import *
from time import gmtime, strftime, time

from tensorflow.keras.callbacks  import EarlyStopping, TensorBoard
from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_recall_fscore_support

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

def train(config_file, step='pretraining'):
    '''
    pretraining/finetuning pipeline
    '''
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    create_target_directory(config_file=config_file,
                            path=config[step]['exp_path'])

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

    # Get callbacks
    cbks = get_callbacks(config, step=step, monitor='val_loss')

    # Loading data
    data = load_pt_data(config, subsets=['train', 'val', 'test'], step=step)

    # Train ASTROMER
    _ = astromer.fit(data['train'],
                  epochs=config[step]['epochs'],
                  validation_data=data['val'],
                  callbacks=cbks)

    # Getting metrics
    loss, r2 = astromer.evaluate(data['test'])
    metrics = {'rmse':loss, 'r_square':r2}
    save_metrics(metrics,
                 path=os.path.join(config[step]['exp_path'],
                                   'metrics.csv'))

def classify(config_file):
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    create_target_directory(config_file=config_file,
                            path=config['classification']['exp_path'])

    # Load data for classification
    data = load_clf_data(config)

    for clf_name in ['mlp_att', 'lstm_att', 'lstm']:
        print('[INFO] Training {}'.format(clf_name))
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
        
        # Create classifier
        clf_model = get_classifier_by_name(clf_name,
                    config,
                    astromer=astromer,
                    train_astromer=config['classification']['train_astromer'])

        # Compile and train
        optimizer = Adam(learning_rate=config['classification']['lr'])
        exp_path_clf = config['classification']['exp_path']
        os.makedirs(exp_path_clf, exist_ok=True)

        clf_model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics='accuracy')

        cbks = get_callbacks(config, step='classification',
                             monitor='val_loss', extra=clf_name)

        history = clf_model.fit(data['train'],
                                epochs=config['classification']['epochs'],
                                callbacks=cbks,
                                validation_data=data['val'])

        clf_model.save(os.path.join(exp_path_clf, clf_name, 'model'))

        # Evaluate
        y_pred = clf_model.predict(data['test'])
        y_true = tf.concat([y for _, y in data['test']], 0)

        pred_labels = tf.argmax(y_pred, 1)
        true_labels = tf.argmax(y_true, 1)

        p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                     pred_labels,
                                                     average='macro')
        metrics = {'precision':p, 'recall':r, 'f1': f,
                   'val_acc': tf.reduce_max(history.history['val_accuracy']).numpy(),
                   'val_loss': tf.reduce_min(history.history['val_loss']).numpy(),
                   'model':clf_name}
        # # Save metrics
        save_metrics(metrics, path=os.path.join(exp_path_clf, 'metrics.csv'))


if __name__ == '__main__':

    directory = sys.argv[1]
    mode = sys.argv[3] # pretraining - finetuning - classification 
    print('[INFO] Mode: {}'.format(mode))  
    if not os.path.isdir(directory):
        directory = [directory]
        
    for config_file in os.listdir(directory):
        if config_file.split('.')[3] != 'b': continue
        if mode == 'classification':
            classify(os.path.join(directory, config_file))
        else:
            train(os.path.join(directory, config_file), step=mode)

