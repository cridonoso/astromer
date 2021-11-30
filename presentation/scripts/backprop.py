import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
import h5py
import os

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

from core.data  import pretraining_records
from core.astromer import get_ASTROMER

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def build_lstm(maxlen, n_classes):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}
    mask = tf.cast(1.-placeholder['mask_in'][...,0], dtype=tf.bool)

    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    x = (x - x_min)/(x_max-x_min)

    x = LSTM(256, dropout=.2, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(256, dropout=.2)(x, mask=mask)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

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
    x = LSTM(256, dropout=.2, return_sequences=True)(x, mask=mask)
    x = LayerNormalization()(x)
    x = LSTM(256, dropout=.2)(x, mask=mask)
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

    x_mean = tf.expand_dims(tf.reduce_mean(x, 1), 1)
    x_std = tf.expand_dims(tf.math.reduce_std(x, 1), 1)
    x = (x - x_mean)/x_std

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    # Loading data
    num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]

    train_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                        opt.batch_size, max_obs=opt.max_obs,
                                        msk_frac=0., rnd_frac=0., same_frac=0.,
                                        sampling=False, shuffle=True,
                                        n_classes=num_cls)

    val_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                      opt.batch_size, max_obs=opt.max_obs,
                                      msk_frac=0., rnd_frac=0., same_frac=0.,
                                      sampling=False, shuffle=False,
                                      n_classes=num_cls)

    conf_file = os.path.join(opt.w, 'conf.json')
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

    weights_path = '{}/weights'.format(opt.w)
    model.load_weights(weights_path)

    if opt.mode == 0:
        model = build_lstm_att(model,
                               maxlen=opt.max_obs,
                               n_classes=num_cls,
                               train_astromer=False)
        target_dir = os.path.join(opt.p, 'lstm_att')
    if opt.mode == 1:
        model = build_mlp_att(model,
                              maxlen=opt.max_obs,
                              n_classes=num_cls,
                              train_astromer=False)
        target_dir = os.path.join(opt.p, 'mlp_att')

    if opt.mode == 2:
        model = build_lstm(maxlen=opt.max_obs,
                           n_classes=num_cls)
        target_dir = os.path.join(opt.p, 'lstm')


    optimizer = Adam(learning_rate=opt.lr)

    os.makedirs(target_dir, exist_ok=True)

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

    _ = model.fit(train_batches,
                  epochs=opt.epochs,
                  batch_size=opt.batch_size,
                  callbacks=[estop, tb],
                  validation_data=val_batches)

    model.save(os.path.join(target_dir, 'model'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    # ASTROMER
    parser.add_argument('--w', default="./runs/huge", type=str,
                        help='ASTROMER pretrained weights')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/alcock', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')

    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=30, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    # RNN HIPERPARAMETERS
    parser.add_argument('--mode', default=0, type=int,
                        help='Classifier model: 0: LSTM + ATT - 1: MLP + ATT - 2 LSTM')

    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU number to be used')

    opt = parser.parse_args()
    run(opt)
