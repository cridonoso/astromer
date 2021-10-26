import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import os

from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, InputLayer, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import Model, Input

from core.metrics import custom_acc
from core.losses import custom_cce
from core.astromer import get_ASTROMER
from core.data import load_records_v3

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def get_mlp(num_classes, encoder, maxlen=200):
    ''' FC + ATT'''
    inputs = {
    'input': Input(shape=(maxlen, 1), name='input'),
    'times': Input(shape=(maxlen, 1), name='times'),
    'mask_in': Input(shape=(maxlen, 1), name='mask'),
    }
    m = tf.cast(1.-inputs['mask_in'][...,0], tf.bool)
    x = encoder(inputs)
    x = tf.ragged.boolean_mask(x, m)
    x = tf.reduce_mean(x, 1)
    x = (x - tf.reduce_mean(x, 0))/tf.math.reduce_std(x, 0)

    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x, name="MLP_ATT")

def get_lstm(units, num_classes, maxlen, dropout=0.5):
    ''' LSTM + LSTM + FC'''
    inputs = {
    'input': Input(shape=(maxlen, 1), name='input'),
    'times': Input(shape=(maxlen, 1), name='times'),
    'mask_in': Input(shape=(maxlen, 1), name='mask'),
    }
    m = tf.cast(1.-inputs['mask_in'][...,0], tf.bool)
    x = tf.concat([inputs['times'],
                   inputs['input']], 2)

    rnn_0 = tf.keras.layers.LSTMCell(256,
                                     recurrent_initializer='zeros',
                                     dropout=dropout,
                                     name='RNN0')
    rnn_1 = tf.keras.layers.LSTMCell(256,
                                     recurrent_initializer='zeros',
                                     dropout=dropout,
                                     name='RNN1')
    stacked = tf.keras.layers.RNN([rnn_0, rnn_1],
                                   return_sequences=True)
    x = stacked(x, mask=m)
    x = BatchNormalization()(x[:, -1, :])
    x = Dense(num_classes, activation='softmax', name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="LSTM")

def get_lstm_att(units, num_classes, encoder, maxlen=200, dropout=0.5):
    ''' ATT + LSTM + LSTM + FC'''
    inputs = {
    'input': Input(shape=(maxlen, 1), name='input'),
    'times': Input(shape=(maxlen, 1), name='times'),
    'mask_in': Input(shape=(maxlen, 1), name='mask'),
    }
    m = tf.cast(1.-inputs['mask_in'][...,0], tf.bool)
    x = encoder(inputs)
    x = (x - tf.expand_dims(tf.reduce_mean(x, 1), 1))/tf.expand_dims(tf.math.reduce_std(x, 1), 1)

    rnn_0 = tf.keras.layers.LSTMCell(256,
                                     recurrent_initializer='zeros',
                                     dropout=dropout,
                                     name='RNN0')
    rnn_1 = tf.keras.layers.LSTMCell(256,
                                     recurrent_initializer='zeros',
                                     dropout=dropout,
                                     name='RNN1')
    stacked = tf.keras.layers.RNN([rnn_0, rnn_1],
                                   return_sequences=True)

    x = stacked(x, mask=m)
    x = BatchNormalization()(x[:, -1, :])
    x = Dense(num_classes, activation='softmax', name='FCN')(x)
    return Model(inputs=inputs, outputs=x, name="LSTM_ATT")

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
    df = pd.read_csv(os.path.join(opt.data, 'test_objs.csv'))
    num_cls = df['class'].unique().size

    train_batches = load_records_v3(os.path.join(opt.data, 'train'),
                                 opt.batch_size,
                                 max_obs=200,
                                 repeat=opt.repeat,
                                 is_train=True,
                                 num_cls=num_cls)

    val_batches = load_records_v3(os.path.join(opt.data, 'val'),
                               opt.batch_size,
                               max_obs=200,
                               repeat=opt.repeat,
                               is_train=True,
                               num_cls=num_cls)



    exp_path = opt.p
    if opt.mode == 'mlp_att':
        encoder = init_astromer(opt.emb)
        model = get_mlp(num_cls, encoder)
        exp_path = os.path.join(opt.p, 'mlp_att')
        print(model.summary())

    if opt.mode == 'lstm':
        model = get_lstm(256, num_cls, 200, dropout=0.5)
        exp_path = os.path.join(opt.p, 'lstm')
        print(model.summary())

    if opt.mode == 'lstm_att':
        encoder = init_astromer(opt.emb)
        model = get_lstm_att(256, num_cls, encoder=encoder, dropout=0.5)
        exp_path = os.path.join(opt.p, 'lstm_att')
        print(model.summary())


    optimizer = Adam(learning_rate=opt.lr)
    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    ckpts = ModelCheckpoint(
        filepath=os.path.join(exp_path, 'ckpt'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    estop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=opt.patience,
        mode='auto',
        restore_best_weights=True)

    tboad = TensorBoard(log_dir='{}/logs'.format(exp_path),
                        histogram_freq=0,
                        write_graph=False)

    hist = model.fit(train_batches,
                     epochs=opt.epochs,
                     callbacks=[ckpts, estop, tboad],
                     validation_data=val_batches,
                     verbose=1)

    print('Saving Model')
    # model.save(os.path.join(exp_path, 'model.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/alcock/', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--emb', default='./weights/astromer_10022021', type=str,
                        help='ASTROMER weights')
    parser.add_argument('--p', default="./experiments/debug/", type=str,
                        help='folder for saving embeddings')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of Epochs')
    parser.add_argument('--patience', default=20, type=int,
                        help='patience for early stopping')
    parser.add_argument('--repeat', default=1, type=int,
                        help='repeat dataset samples')
    parser.add_argument('--mode', default='mlp_att', type=str,
                        help='mlp_att - lstm - lstm_att')

    opt = parser.parse_args()
    run(opt)
