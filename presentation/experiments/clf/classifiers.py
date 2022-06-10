import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
import os

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras import Input, Model


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings\

def normalize_batch(tensor):
    min_ = tf.expand_dims(tf.reduce_min(tensor, 1), 1)
    max_ = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
    tensor = tf.math.divide_no_nan(tensor - min_, max_ - min_)
    return tensor

class NormedLSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = ((self.units, self.units), (self.units, self.units))

        super(NormedLSTMCell, self).__init__(**kwargs)

        self.cell_0 = tf.keras.layers.LSTMCell(self.units)
        self.cell_1 = tf.keras.layers.LSTMCell(self.units)
        self.bn = LayerNormalization(name='bn_step')

    def call(self, inputs, states, training=False):
        s0, s1 = states[0], states[1]
        output, s0 = self.cell_0(inputs, states=s0, training=training)
        output = self.bn(output, training=training)
        output, s1 = self.cell_1(output, states=s1, training=training)
        return output, [s0, s1]
    
    def get_config(self):
        config = super(NormedLSTMCell, self).get_config().copy()
        config.update({"units": self.units})
        return config

    
def build_lstm(maxlen, n_classes, state_dim=256):
    print('[INFO] Building LSTM Baseline')
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}
    
    m = tf.cast(1.-placeholder['mask_in'][...,0], tf.bool)
    tim = normalize_batch(placeholder['times'])
    inp = normalize_batch(placeholder['input'])
    x = tf.concat([tim, inp], 2)

    cell_0 = NormedLSTMCell(units=state_dim)
    dense  = Dense(n_classes, name='FCN')

    s0 = [tf.zeros([tf.shape(x)[0], state_dim]),
          tf.zeros([tf.shape(x)[0], state_dim])]
    s1 = [tf.zeros([tf.shape(x)[0], state_dim]),
          tf.zeros([tf.shape(x)[0], state_dim])]

    rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
    x = rnn(x, initial_state=[s0, s1], mask=m)
    x = tf.nn.dropout(x, .3)
    x = dense(x)
    return Model(placeholder, outputs=x, name="LSTM")

def build_lstm_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = tf.cast(1.-placeholder['mask_in'][...,0], dtype=tf.bool)

    x = encoder(placeholder, training=False)
    x = tf.math.divide_no_nan(x-tf.expand_dims(tf.reduce_mean(x, 1),1),
                              tf.expand_dims(tf.math.reduce_std(x, 1), 1))        
    x = LSTM(256, dropout=.3, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(256, dropout=.3)(x, mask=mask)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def build_mlp_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=False)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")



