import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import os

from tensorflow.keras.layers import (LSTM, Dense,
                                     BatchNormalization,
                                     LayerNormalization,
                                     Input, Dropout)
from tensorflow.keras import Model
from tqdm import tqdm
from core.metrics import custom_acc
from core.losses import custom_bce
from core.astromer import get_ASTROMER
from core.data import load_records
from core.tboard import save_scalar

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def get_mlp(num_classes, encoder, maxlen=200):
    ''' FC + ATT'''
    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}
    x = encoder(placeholder)
    x = tf.reduce_mean(x, 1)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="MLP")

def get_lstm(units, num_classes, maxlen, dropout=0.5):
    ''' LSTM + LSTM + FC'''

    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen),
                  batch_size=None,
                  name='mask')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    bool_mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))

    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=placeholder, outputs=x, name="RNNCLF")

def get_lstm_att(units, num_classes, encoder, maxlen=200, dropout=0.5):
    ''' ATT + LSTM + LSTM + FC'''

    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}
    x = encoder(placeholder)

    bool_mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=placeholder, outputs=x, name="RNNCLF")

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        y_pred = model(batch)
        ce = custom_bce(y_true=batch['label'], y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    grads = tape.gradient(ce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return acc, ce

@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        y_pred = model(batch, training=False)
        ce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return acc, ce, y_pred, batch['label']
    return acc, ce

def init_astromer(path):
    conf_file = os.path.join(path, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'],
                         use_leak  =conf['use_leak'])

    weights_path = '{}/weights'.format(path)
    model.load_weights(weights_path)
    model.trainable=False
    encoder = model.get_layer('encoder')
    return encoder

def run(opt):
    optimizer = tf.keras.optimizers.Adam(1e-3)
    train_cce  = tf.keras.metrics.Mean(name='train_bce')
    valid_cce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    train_dataset, val_dataset = load_records('./data/records/ogle/train',
                                              opt.batch_size,
                                              val_data=0.1,
                                              no_shuffle=False,
                                              max_obs=200,
                                              msk_frac=0.,
                                              rnd_frac=0.,
                                              same_frac=0.,
                                              repeat=1)
    num_classes = 10
    for x in train_dataset:
        num_classes = np.unique(x['label']).shape[0]
        break

    exp_path = opt.p
    if opt.mode == 'mlp_att':
        encoder = init_astromer(opt.emb)
        model = get_mlp(num_classes, encoder)
        exp_path = os.path.join(opt.p, 'mlp_att')

    if opt.mode == 'lstm':
        model = get_lstm(256, num_classes, 200, dropout=0.5)
        exp_path = os.path.join(opt.p, 'lstm')

    if opt.mode == 'lstm_att':
        encoder = init_astromer(opt.emb)
        model = get_lstm_att(256, num_classes, encoder, dropout=0.5)
        exp_path = os.path.join(opt.p, 'lstm_att')

    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'valid'))

    best_loss = 999999.
    es_count = 0
    pbar = tqdm(range(1000), desc='epoch')
    for epoch in pbar:
        for batch in tqdm(train_dataset, desc='iterations'):
            acc, cce = train_step(model, batch, optimizer)
            train_acc.update_state(acc)
            train_cce.update_state(cce)

        for batch in val_dataset:
            val_acc, val_cce = valid_step(model, batch)
            valid_acc.update_state(val_acc)
            valid_cce.update_state(val_cce)

        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_cce, epoch, name='xentropy')
        save_scalar(valid_writter, valid_cce, epoch, name='xentropy')

        if valid_cce.result() < best_loss:
            best_loss = valid_cce.result()
            es_count = 0.
            model.save_weights(os.path.join(exp_path, 'weights'))
        else:
            es_count+=1.

        if es_count == opt.patience:
            print('[INFO] Early Stopping Triggered')
            break

        msg = 'EPOCH {} - ES COUNT: {}/{} Train acc: {:.4f} - Val acc: {:.4f} - Train CE: {:.2f} - Val CE: {:.2f}'.format(
                                                                                      epoch,
                                                                                      es_count,
                                                                                      opt.patience,
                                                                                      train_acc.result(),
                                                                                      valid_acc.result(),
                                                                                      train_cce.result(),
                                                                                      valid_cce.result())
        pbar.set_description(msg)
        valid_cce.reset_states()
        train_cce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--emb', default='./embeddings/ogle_20/train', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./experiments/exp_1/", type=str,
                        help='folder for saving embeddings')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--valptg', default=0.2, type=float,
                        help='validation subset fraction')
    parser.add_argument('--patience', default=20, type=int,
                        help='patience for early stopping')

    parser.add_argument('--mode', default='mlp', type=str,
                        help='mlp_att - lstm - lstm_att')

    opt = parser.parse_args()
    run(opt)
