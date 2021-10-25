import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse
import pickle
import json
import os

from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, InputLayer, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import Model, Input

from core.data import load_records, load_records_v2, load_embeddings
from sklearn.metrics import confusion_matrix, accuracy_score

def get_lstm(num_cls):
    x = Input(shape=(200, 2), name='x')
    m = Input(shape=(200, ), name='mask', dtype='bool')
    inputs={'x': x, 'mask': m}
    x = LSTM(units=256, return_sequences=True,  name='lstm_0',
             dropout=0.5)(inputs['x'], mask=inputs['mask'])
    x = BatchNormalization(name='bn_0')(x)
    x = LSTM(units=256, return_sequences=False,  name='lstm_1',
             dropout=0.5)(x, mask=inputs['mask'])
    x = BatchNormalization(name='bn_1')(x)
    y = Dense(num_cls, name='dense')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=y, name='LSTM')
    return model

def get_mlp(num_cls):
    model = tf.keras.Sequential(name='MLP_ATT')
    model.add(Input(shape=(256,), name='x'))
    model.add(Input(shape=(200,), name='mask'))
    model.add(BatchNormalization(name='bn0'))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization(name='bn3'))
    model.add(Dense(num_cls))
    return model

def get_lstm_att(num_cls):
    x = Input(shape=(200, 256), name='x')
    m = Input(shape=(200, ), name='mask', dtype='bool')
    inputs={'x': x, 'mask': m}
    x = LSTM(units=256, return_sequences=True,  name='lstm_0',
             dropout=0.5)(inputs['x'], mask=inputs['mask'])
    x = BatchNormalization(name='bn_0')(x)
    x = LSTM(units=256, return_sequences=False,  name='lstm_1',
             dropout=0.5)(x, mask=inputs['mask'])
    x = BatchNormalization(name='bn_1')(x)
    y = Dense(num_cls, name='dense')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=y, name='LSTM_ATT')
    return model

def run(opt):

    test_objs = pd.read_csv('{}/test_objs.csv'.format(opt.data))
    num_cls   = test_objs['class'].unique().size

    if opt.mode == 'lstm':
        train_batches = load_records_v2('{}/train'.format(opt.data), opt.batch_size,
                                max_obs=200,
                                msk_frac=0.,
                                rnd_frac=0.,
                                same_frac=0.,
                                repeat=5,
                                is_train=True)

        val_batches = load_records_v2('{}/val'.format(opt.data), opt.batch_size,
                                      max_obs=200,
                                      msk_frac=0.,
                                      rnd_frac=0.,
                                      same_frac=0.,
                                      repeat=5,
                                      is_train=True)
    else:
        if opt.mode == 'mlp_att':
            time_avg = True
        else:
            time_avg = False

        train_batches = load_embeddings('{}/train'.format(opt.data),
                                        num_cls,
                                        opt.batch_size,
                                        is_train=True,
                                        time_avg=time_avg)
        val_batches   = load_embeddings('{}/val'.format(opt.data),
                                        num_cls,
                                        opt.batch_size,
                                        is_train=True,
                                        time_avg=time_avg)



    if opt.mode == 'lstm':
        model = get_lstm(num_cls)
        exp_path = '{}/lstm'.format(opt.p)
    if opt.mode == 'mlp_att':
        model = get_mlp(num_cls)
        exp_path = '{}/mlp_att'.format(opt.p)
    if opt.mode == 'lstm_att':
        model = get_lstm_att(num_cls)
        exp_path = '{}/lstm_att'.format(opt.p)

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    estop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=50, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    tboad = TensorBoard(log_dir='{}/logs'.format(exp_path), histogram_freq=0, write_graph=False)

    hist = model.fit(train_batches,
                     epochs=opt.epochs,
                     callbacks=[estop, tboad],
                     validation_data=val_batches,
                     verbose=0)

    model.save(os.path.join(exp_path, 'model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='folder for saving embeddings')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='num of epochs')
    parser.add_argument('--mode', default="lstm_att", type=str,
                        help='mlp_att - lstm_att - lstm')

    opt = parser.parse_args()
    run(opt)
