import tensorflow as tf
import numpy as np
import argparse
import optuna
import h5py
import os

from tensorflow.keras.layers import BatchNormalization, \
                                    Dense, \
                                    LSTM, \
                                    LayerNormalization, \
                                    Normalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import os


def load_embeddings(source):
    file = open(source, 'rb')
    hf = h5py.File(file)
    att = hf['att'][()]
    x = hf['x'][()]
    t = hf['t'][()]
    lc = np.concatenate([t, x], 2)

    y = hf['y'][()]
    l = hf['id'][()]
    m = 1. - hf['m'][()]
    return att, y, l, m, lc

def create_lstm(trial, n_classes):
    # 2. Suggest values of the hyperparameters using a trial object.

    inputs = tf.keras.Input(shape=(200, 2), name='input')
    mask = tf.keras.Input(shape=(200, ), dtype=tf.bool, name='mask')

    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    x = (inputs - x_mean)/x_std
    
    units_list_0 = trial.suggest_int('units_0', 16, 512)
    drop_list_0  = trial.suggest_float("dropout_0", 0, 0.5)
    
    units_list_1  = trial.suggest_int('units_1', 16, 512)
    drop_list_1   = trial.suggest_float("dropout_1", 0, 0.5)
    
    x = LSTM(units_list_0, dropout=drop_list_0, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(units_list_1, dropout=drop_list_1)(x, mask=mask)
    
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)

    model = tf.keras.Model(inputs=[inputs, mask], outputs=x)
                   
    return model

def create_lstm_att(trial, n_classes):
    # 2. Suggest values of the hyperparameters using a trial object.

    inputs = tf.keras.Input(shape=(200, 256), name='input')
    mask = tf.keras.Input(shape=(200, ), dtype=tf.bool, name='mask')

    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    x = (inputs - x_mean)/x_std
    
    units_list_0 = trial.suggest_int(f'units_0', 16, 512)
    drop_list_0  = trial.suggest_float("dropout_0", 0, 0.5)
    
    units_list_1  = trial.suggest_int(f'units_1', 16, 512)
    drop_list_1   = trial.suggest_float("dropout_1", 0, 0.5)
    
    x = LSTM(units_list_0, dropout=drop_list_0, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(units_list_1, dropout=drop_list_1)(x, mask=mask)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)

    model = tf.keras.Model(inputs=[inputs, mask], outputs=x)
                   
    return model

def create_mlp(trial, n_classes):
    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 3)
            
    inputs = tf.keras.Input(shape=(256))
    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    
    x = (inputs - x_mean)/x_std
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 16, 2048, log=True)
        x = Dense(num_hidden, activation='relu')(x)
        
    x = Dense(n_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
           
    return model

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["Adam", "RMSprop"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

def learn(model, optimizer, x_train, y_train, x_val, y_val):
    
    model.compile(optimizer=optimizer, 
              loss=CategoricalCrossentropy(from_logits=True), 
                  metrics='accuracy')
    estop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=100, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
        )

    _ = model.fit(x_train, y_train, 
                  epochs=100,
                  batch_size=2048,
                  callbacks=[estop],
                  verbose=1,
                  validation_data=(x_val, y_val))
    
    y_pred = model.predict(x_val)
    y_pred_labs = tf.argmax(y_pred, 1)
    prec, rec, f1, _ = precision_recall_fscore_support(tf.argmax(y_val, 1), y_pred_labs, average='macro')
    return f1

def objective(trial, mode='lstm', datapath='.'):
    # Get data.
    
    x_0, y_0, l_0, m_0, lc_0 = load_embeddings(os.path.join(datapath, 'train.h5'))
    x_1, y_1, l_1, m_1, lc_1 = load_embeddings(os.path.join(datapath, 'val.h5'))
    
    n_classes = len(np.unique(y_0))
    y_train = tf.one_hot(y_0, n_classes)
    y_val = tf.one_hot(y_1, n_classes)
    
    if mode == 'mlp_att':
        print(mode)
        x_train = np.sum(x_0*m_0, 1)/tf.reduce_sum(m_0, 1)
        x_val = np.sum(x_1*m_1, 1)/tf.reduce_sum(m_1, 1)
        model = create_mlp(trial, n_classes)
        
    if mode == 'lstm_att':
        print(mode)
        x_train = (x_0, m_0)
        x_val = (x_1, m_1)
        model = create_lstm_att(trial, n_classes)
        
    if mode == 'lstm':
        print(mode)
        x_train = (lc_0, m_0)
        x_val = (lc_1, m_1)
        model = create_lstm(trial, n_classes)
        
    # Build optimizer.
    optimizer = create_optimizer(trial)
    
    # Training and validating cycle.
    f1 = learn(model, optimizer, x_train, y_train, x_val, y_val)

    # Return last validation accuracy.
    return f1

def wrap_fn(func, mode, datapath):
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs):
        result = func(*args, mode=mode, datapath=datapath)
        return result
    return wrap


def run(opt):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    obj = wrap_fn(objective, mode=opt.mode, datapath=opt.data)

    study = optuna.create_study(direction="maximize", 
                                storage='sqlite:///{}_{}.db'.format(opt.data.split('/')[-3], opt.mode), 
                                study_name='{}_{}'.format(opt.data.split('/')[-3], opt.mode))
    study.optimize(obj, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU number to be used')
    parser.add_argument('--data', default='./data/records/alcock', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--mode', default='mlp', type=str,
                        help='mlp_att - lstm_att - lstm')

    opt = parser.parse_args()
    run(opt)
