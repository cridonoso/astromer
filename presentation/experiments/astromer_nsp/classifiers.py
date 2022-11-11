import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.layers     import Dense, LSTM, LayerNormalization
from tensorflow.keras            import Input, Model

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


def build_lstm(maxlen, n_classes):
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

    cell_0 = NormedLSTMCell(units=256)
    dense  = Dense(n_classes, name='FCN')

    s0 = [tf.zeros([tf.shape(x)[0], 256]),
          tf.zeros([tf.shape(x)[0], 256])]
    s1 = [tf.zeros([tf.shape(x)[0], 256]),
          tf.zeros([tf.shape(x)[0], 256])]

    rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
    x = rnn(x, initial_state=[s0, s1], mask=m)
    x = tf.nn.dropout(x, .3)
    x = dense(x)
    return Model(placeholder, outputs=x, name="LSTM")

def build_lstm_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    print('BUILDING LSTM + ATT')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    cell_0 = NormedLSTMCell(units=256)
    dense  = Dense(n_classes, name='FCN')

    s0 = [tf.zeros([tf.shape(placeholder['input'])[0], 256]),
          tf.zeros([tf.shape(placeholder['input'])[0], 256])]
    s1 = [tf.zeros([tf.shape(placeholder['input'])[0], 256]),
          tf.zeros([tf.shape(placeholder['input'])[0], 256])]
    rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = tf.cast(1.-placeholder['mask_in'][...,0], dtype=tf.bool)
    x = encoder(placeholder, training=train_astromer)
    x = tf.math.divide_no_nan(x-tf.expand_dims(tf.reduce_mean(x, 1),1),
                              tf.expand_dims(tf.math.reduce_std(x, 1), 1))
    x = rnn(x, initial_state=[s0, s1], mask=mask)
    x = tf.nn.dropout(x, .3)
    x = dense(x)
    return Model(placeholder, outputs=x, name="LSTM_ATT")

def build_mlp_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    print('BUILDING MLP + ATT')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=train_astromer)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def build_mlp_att_cls(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    print('BUILDING MLP + ATT')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer
        
    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=train_astromer)
    x = tf.slice(x, [0,0,0], [-1,1,-1])
    
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def get_classifier_by_name(name, config, astromer=None, train_astromer=False):
    num_cls = pd.read_csv(
                os.path.join(config['classification']['data']['path'],
                            'objects.csv')).shape[0]

    if name == 'lstm':
        clf_model= build_lstm(config['astromer']['window_size'],
                              num_cls)

    if name== 'lstm_att':
        clf_model= build_lstm_att(astromer,
                                  config['astromer']['window_size'],
                                  num_cls,
                                  train_astromer=train_astromer)

    if name == 'mlp_att':
        clf_model= build_mlp_att(astromer,
                                 config['astromer']['window_size'],
                                 num_cls,
                                 train_astromer=train_astromer)
    if name == 'mlp_att_cls':
        clf_model= build_mlp_att_cls(astromer,
                                     config['astromer']['window_size'],
                                     num_cls,
                                     train_astromer=train_astromer)
    return clf_model
