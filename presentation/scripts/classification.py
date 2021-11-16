import tensorflow as tf
import pandas as pd
import argparse
import logging
import json
import time
import os

import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def load_embeddings(source):
    file = open(source, 'rb')
    hf = h5py.File(file)
    att = hf['att'][()]
    x = hf['x'][()]
    t = hf['t'][()]
    lc = np.concatenate([t, x], 2)
    y = hf['y'][()]
    m = 1. - hf['m'][()]
    return att, y, m, lc

def build_lstm_att(unit_1=256, unit_2=256, drop_1=0.2, drop_2=0.2, n_classes=5):
    inputs = tf.keras.Input(shape=(200, 256), name='input')
    mask = tf.keras.Input(shape=(200, ), dtype=tf.bool, name='mask')
    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    x = (inputs - x_mean)/x_std
    x = LSTM(unit_1, dropout=drop_1, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(unit_2, dropout=drop_2)(x, mask=mask)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    model = tf.keras.Model(inputs=[inputs, mask], outputs=x)
    return model 

def build_mpl_att(n_layers=3, units=[1024,512,256], n_classes=5):
    inputs = tf.keras.Input(shape=(256))
    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    x = (inputs - x_mean)/x_std
    for i in range(n_layers):
        x = Dense(units[i], activation='relu')(x)
        x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def build_lstm(unit_1=256, unit_2=256, drop_1=0.2, drop_2=0.2, n_classes=5):
    inputs = tf.keras.Input(shape=(200, 2), name='input')
    mask = tf.keras.Input(shape=(200, ), dtype=tf.bool, name='mask')
    
    x_mean = tf.expand_dims(tf.reduce_mean(inputs, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(inputs, 1), 1)
    x = (inputs - x_mean)/x_std
    
    x = LSTM(unit_1, dropout=drop_1, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(unit_2, dropout=drop_2)(x, mask=mask)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    model = tf.keras.Model(inputs=[inputs, mask], outputs=x)
    return model

def run(opt):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    
    # Loading saved embeddings
    x_0, y_0, m_0, lc_0 = load_embeddings(os.path.join(opt.data, 'train.h5'))
    x_1, y_1, m_1, lc_1 = load_embeddings(os.path.join(opt.data, 'val.h5'))
    
    n_classes = len(np.unique(y_0))
    y_train = np.concatenate([y_0, y_1])
    y_train = tf.one_hot(y_train, n_classes) # One-hot encoding
    
    
    optimizer = Adam(learning_rate=opt.lr)        
    # Init. classifier
    if opt.mode == 0: # LSTM + ATT
        # Formatting data
        x_train = np.concatenate([x_0, x_1])
        m_train = np.concatenate([m_0, m_1])
        x_train = [x_train, m_train]
        # Create model
        if os.path.isfile(opt.conf):
            print('[INFO] LOADING PREDEFINED CONFIG')
            with open(opt.conf, 'r') as f:
                config = json.load(f)
            print(config)
            model = build_lstm_att(unit_1=config['units_0'], unit_2=config['units_1'], 
                           drop_1=config['dropout_0'], drop_2=config['dropout_1'], 
                           n_classes=n_classes)
            if config['optimizer'] == 'Adam':
                optimizer = Adam(learning_rate=config['adam_learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=config['rmsprop_learning_rate'])
        else:
            model = build_lstm_att(n_classes=n_classes)
            
        
        target_dir = os.path.join(opt.p, 'lstm_att')
        
    if opt.mode == 1: # MLP + ATT
        # Formatting data
        mean_1 = np.sum(x_0*m_0, 1)/tf.reduce_sum(m_0)
        mean_2 = np.sum(x_1*m_1, 1)/tf.reduce_sum(m_1)
        x_train = np.concatenate([mean_1, mean_2])

        # Create model
        if os.path.isfile(opt.conf):
            print('[INFO] LOADING PREDEFINED CONFIG')
            with open(opt.conf, 'r') as f:
                config = json.load(f) 
            units = [val for key, val in config.items() if 'n_units'in key]
            model = build_mpl_att(n_layers=config['n_layers'], units=units, n_classes=n_classes)
            if config['optimizer'] == 'Adam':
                optimizer = Adam(learning_rate=config['adam_learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=config['rmsprop_learning_rate'])
        else:
            model = build_mpl_att(n_classes=n_classes)
            
        
        target_dir = os.path.join(opt.p, 'mlp_att')
        
        
    if opt.mode == 2: # LSTM
        # Formatting data
        x_train = np.concatenate([lc_0, lc_1])
        m_train = np.concatenate([m_0, m_1])
        x_train = [x_train, m_train]
        
        # Create model
        if os.path.isfile(opt.conf):
            print('[INFO] LOADING PREDEFINED CONFIG')
            with open(opt.conf, 'r') as f:
                config = json.load(f)
            print(config)
            model = build_lstm(unit_1=config['units_0'], unit_2=config['units_1'], 
                                   drop_1=config['dropout_0'], drop_2=config['dropout_1'], 
                                   n_classes=n_classes)
            if config['optimizer'] == 'Adam':
                optimizer = Adam(learning_rate=config['adam_learning_rate'])
            else:
                optimizer = RMSprop(learning_rate=config['rmsprop_learning_rate'])
        else:
            model = build_lstm(n_classes=n_classes)
            


        target_dir = os.path.join(opt.p, 'lstm')
        
    # Creating (--p)royect directory
    os.makedirs(opt.p, exist_ok=True)

    model.compile(optimizer=optimizer, 
                  loss=CategoricalCrossentropy(from_logits=True), 
                  metrics='accuracy')
    
    estop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=opt.patience, 
                          verbose=0,
                          mode='auto', 
                          baseline=None, 
                          restore_best_weights=True)
    tb = TensorBoard(log_dir=os.path.join(target_dir, 'logs'), 
                     write_graph=False,
                     write_images=False, 
                     write_steps_per_second=False, 
                     update_freq='epoch',
                     profile_batch=0, 
                     embeddings_freq=0, 
                     embeddings_metadata=None)
    
    _ = model.fit(x_train, y_train, 
                  epochs=opt.epochs,
                  batch_size=opt.batch_size,
                  callbacks=[estop, tb],
                  validation_split=0.2)
    
    model.save(os.path.join(target_dir, 'model'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU number to be used')
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=2048, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=100, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')
    parser.add_argument('--mode', default=0, type=int,
                        help='Classifier model: 0: LSTM + ATT - 1: MLP + ATT - 2 LSTM')
    
    parser.add_argument('--conf', default='./hypersearch/something.json', type=str,
                        help='Hyperparameter configuration')
    

    opt = parser.parse_args()
    run(opt)
