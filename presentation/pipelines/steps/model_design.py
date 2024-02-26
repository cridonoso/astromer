import tensorflow as tf 
import toml 
import os

from src.models.astromer_0 import get_ASTROMER as get_Base
from src.models.astromer_nsp import get_ASTROMER as get_NSP
from src.models.astromer_skip import get_ASTROMER as get_Skip

from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Dense, Dropout


def build_model(params):
    if params['arch'] == 'normal' or params['arch'] == 'base':
        model = get_Base(num_layers=params['num_layers'],
                             d_model=params['head_dim']*params['num_heads'],
                             num_heads=params['num_heads'],
                             dff=params['mixer'],
                             base=params['pe_base'],
                             rate=params['dropout'],
                             use_leak=False,
                             maxlen=params['window_size'],
                             m_alpha=params['m_alpha'])

    if params['arch'] == 'skip':
        model = get_Skip(num_layers=params['num_layers'],
                         num_heads=params['num_heads'],
                         head_dim=params['head_dim'],
                         mixer_size=params['mixer'],
                         dropout=params['dropout'],
                         pe_base=params['pe_base'],
                         pe_dim=params['pe_dim'],
                         pe_c=params['pe_exp'],
                         window_size=params['window_size'],
                         m_alpha=params['m_alpha'],
                         mask_format=params['mask_format'])

    if params['arch'] == 'nsp':
        model = get_NSP(num_layers=params['num_layers'],
                        num_heads=params['num_heads'],
                        head_dim=params['head_dim'],
                        mixer_size=params['mixer'],
                        dropout=params['dropout'],
                        pe_base=params['pe_base'],
                        pe_dim=params['pe_dim'],
                        pe_c=params['pe_exp'],
                        window_size=params['window_size'],
                        m_alpha=params['m_alpha'],
                        mask_format=params['mask_format'])

    return model

def load_pt_model(pt_path):
    config_file = os.path.join(pt_path, 'config.toml')
    with open(config_file, 'r') as file:
        pt_config = toml.load(file)
    model = build_model(pt_config)
    weights_path = os.path.join(pt_path, 'weights')
    model.load_weights(weights_path).expect_partial()
    return model, pt_config


def get_mlp_avg(inputs, mask, num_cls):
    x = tf.multiply(inputs, mask) 
    x = tf.reduce_sum(x, 1)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, 1))

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(num_cls, name='output_layer')(x)
    return y_pred


def get_mlp_att(inputs, mask, num_cls):
    x = TimeDistributed(Dense(1024, activation='relu'))(inputs)
    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = TimeDistributed(Dense(num_cls, name='output_layer'))(x)
    y_pred = tf.reduce_sum(y_pred*mask,1)
    y_pred = tf.math.divide_no_nan(y_pred, tf.reduce_sum(mask, 1))
    return y_pred

def get_linear(inputs, mask, num_cls):
    y_pred = TimeDistributed(Dense(num_cls, name='output_layer'))(inputs)
    y_pred = tf.reduce_sum(y_pred*mask,1)
    y_pred = tf.math.divide_no_nan(y_pred, mask)
    return y_pred