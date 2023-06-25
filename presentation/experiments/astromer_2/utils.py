import tensorflow as tf 
import pandas as pd 
import wandb
import os

from src.layers.custom_rnn import NormedLSTMCell, build_zero_init_state 
from src.data import pretraining_pipeline
from src.models import build_input_2

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.layers     import Dense, Conv1D, Flatten, RNN, LSTM, LayerNormalization
from tensorflow.keras            import Input, Model
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support

def classify(clf_model, data, exp_path_clf, lr=1e-5, model_name='mlp', debug=False):
    
    if debug:
        for k in data.keys():
            data[k] = data[k].take(1) 
    
    # Compile and train
    optimizer = Adam(learning_rate=lr)
    os.makedirs(exp_path_clf, exist_ok=True)

    clf_model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics='accuracy')

    cbks = get_callbacks(os.path.join(exp_path_clf, clf_model.name), monitor='val_loss')
    
    history = clf_model.fit(data['train'],
                            epochs=1 if debug else 100000,
                            callbacks=cbks,
                            validation_data=data['val'])

    clf_model.save_weights(os.path.join(exp_path_clf, clf_model.name, 'model'))

    y_pred = clf_model.predict(data['test'])
    y_true = tf.concat([y for _, y in data['test']], 0)

    pred_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y_true, 1)

    p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                 pred_labels,
                                                 average='macro',
                                                 zero_division=0.)
    
    metrics = {'test_precision':p, 
               'test_recall':r, 
               'test_f1': f}

    return clf_model, metrics

def create_classifier(astromer, config, num_cls=None, train_astromer=False, name='mlp_att'):
    placeholder = build_input_2(config.window_size+3)

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer
    
    z_dim = config.head_dim*config.n_heads

    x = encoder(placeholder, training=train_astromer)
    
    if name == 'lstm':
        print('[INFO] Training an MLP on light curves directly')
        m = tf.cast(1.-placeholder['mask_in'][...,0], tf.bool)
        tim = normalize_batch(placeholder['times'])
        inp = normalize_batch(placeholder['input'])
        x = tf.concat([tim, inp], 2)

        cell_0 = NormedLSTMCell(units=256)
        zero_state = build_zero_init_state(x, 256)
        rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
        drop_layer = tf.keras.layers.Dropout(.3)
        
        x = rnn(x, initial_state=zero_state, mask=m)
        x = drop_layer(x)
            
    if name == 'mlp_att':
        print('[INFO] Training an MLP on time-mean Z')
        mask = placeholder['mask_out']
        x = x * mask
        x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = LayerNormalization()(x)
    
    if name == 'mlp_cls':
        x = tf.slice(x, [0,0,0], [-1,1,-1], name='cls_tkn')
        x = tf.reshape(x, [-1, z_dim])
        
    if name == 'mlp_att_lite':
        print('[INFO] Training an MLP on time-mean Z')
        mask = placeholder['mask_out']
        x = x * mask
        x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
        
    x = Dense(num_cls)(x)
    return Model(inputs=placeholder, outputs=x, name=name)

def get_batch_size(model, bytes_per_param=4, window_size=None):
    params = model.count_params()    
    if window_size > 200:
        bs = int(300*595841/params)
    else:
        bs = int(3000*595841/params)
    return min(bs, 3000)

def load_clf_data(config, batch_size, num_cls, datapath=None, debug=False):
    if datapath is None:
        datapath = config.pt_data
        
    data = dict()
    for subset in ['train', 'val', 'test']:
        data[subset] = pretraining_pipeline(
                os.path.join(datapath, subset),
                batch_size,
                config.window_size,
                0.,
                0.,
                0.,
                False,
                True,
                num_cls=num_cls,
                normalize='zero-mean',
                cache=True,
                nsp_prob=config.nsp_prob,
                nsp_frac=config.nsp_fraction,
                moving_window=False)
        if debug:
            data[subset] = data[subset].take(1)

    return data

def load_pt_data(config, sampling=True, datapath=None, debug=False):
    
    if datapath is None:
        datapath = config.pt_data
        
    data = dict()
    for subset in ['train', 'val', 'test']:
        repeat = 4 if subset == 'train' else None
        data[subset] = pretraining_pipeline(os.path.join(datapath, subset),
                                            config.batch_size,
                                            config.window_size,
                                            config.probed,
                                            config.rand,
                                            config.rand,
                                            sampling,
                                            True,
                                            repeat=repeat,
                                            num_cls=None,
                                            normalize='zero-mean',
                                            cache=True,
                                            nsp_prob=config.nsp_prob,
                                            nsp_frac=config.nsp_fraction,
                                            moving_window=False)
        if debug:
            data[subset] = data[subset].take(1)
    return data



def get_callbacks(path, monitor='val_loss', extra=''):
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(path, extra, 'weights'),
            save_weights_only=True,
            monitor=monitor,
            save_best_only=True),
        EarlyStopping(monitor=monitor,
            patience = 20,
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(path, extra, 'logs'),
            histogram_freq=1,
            write_graph=True),
        WandbMetricsLogger(log_freq='epoch')]
    return callbacks

def check_if_exist_finetuned_weights(config, project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.config['subdataset'] == config.subdataset and \
           run.config['fold']== config.fold and \
           run.state == 'finished':
            return True
    return False