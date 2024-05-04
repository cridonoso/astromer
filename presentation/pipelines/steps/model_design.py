import tensorflow as tf 
import toml 
import os

from src.models.astromer_0 import get_ASTROMER as get_Zero
from src.models.astromer_1 import get_ASTROMER as get_Base
from src.models.astromer_nsp import get_ASTROMER as get_NSP
from src.models.astromer_skip import get_ASTROMER as get_Skip

from src.models.astromer_nsp import  build_input as build_input_nsp
from src.models.astromer_skip import  build_input as build_input_skip
from src.models.astromer_1 import build_input as build_input_base
from src.models.astromer_0 import build_input as build_input_zero


from tensorflow.keras import Model
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Dense, Dropout


def build_model(params, return_weights=False):
    if not 'mask_format' in params.keys():
        params['mask_format'] = None
        
    if 'temperature' not in params.keys():
        params['temperature'] = 0.
        
    if params['arch'] == 'zero':
        print('[INFO] Zero architecture loaded')
        model = get_Zero(num_layers=params['num_layers'],
                           d_model=params['head_dim']*params['num_heads'],
                           num_heads=params['num_heads'],
                           dff=params['mixer'],
                           base=params['pe_base'],
                           rate=params['dropout'],
                           use_leak=False,
                           maxlen=params['window_size'],
                           m_alpha=params['m_alpha'],
                           mask_format=params['mask_format'],
                           return_weights=return_weights,
                           loss_format=params['loss_format'],
                           correct_loss=params['correct_loss'],
                           temperature=params['temperature'])

    if params['arch'] == 'base':
        model = get_Base(num_layers=params['num_layers'],
                          num_heads=params['num_heads'],
                          head_dim=params['head_dim'],
                          mixer_size=params['mixer'],
                          dropout=params['dropout'],
                          pe_base=params['pe_base'],
                          pe_dim=params['pe_dim'],
                          pe_c=params['pe_exp'],
                          window_size=params['window_size'],
                          m_alpha=params['m_alpha'],
                          mask_format=params['mask_format'],
                          use_leak=params['use_leak'],
                          loss_format=params['loss_format'],
                          correct_loss=params['correct_loss'],
                          temperature=params['temperature'])

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

def load_pt_model(pt_path, optimizer=None):
    config_file = os.path.join(pt_path, 'config.toml')
    with open(config_file, 'r') as file:
        pt_config = toml.load(file)
    model = build_model(pt_config)
    weights_path = os.path.join(pt_path, 'weights')
    if optimizer is not None:
        model.compile(optimizer=optimizer)
    model.load_weights(weights_path).expect_partial()
    return model, pt_config


def get_avg_mlp(inputs, mask, num_cls):
    x = tf.multiply(inputs, mask) 
    x = tf.reduce_sum(x, 1)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, 1))

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(num_cls, name='output_layer')(x)
    return y_pred

def get_avg_mlp_dp(inputs, mask, num_cls):
    x = tf.multiply(inputs, mask) 
    x = tf.reduce_sum(x, 1)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, 1))

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(num_cls, name='output_layer')(x)
    return y_pred

def get_mlp_avg(inputs, mask, num_cls):
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

def get_skip_avg_mlp(inputs, mask, num_cls):

    x = tf.stack(inputs, axis=0)
    mask = tf.expand_dims(mask, axis=0)
    x = tf.multiply(x, mask) 
    x = tf.reduce_sum(x, 2)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, 2))

    initial_value = tf.random.truncated_normal([len(inputs)])
    gamma = tf.Variable(initial_value=initial_value,
                        trainable=True)
    gamma = tf.keras.layers.Softmax()(gamma)
    gamma = tf.reshape(gamma, [len(inputs), 1, 1])
    x = tf.multiply(x, gamma)
    x = tf.reduce_mean(x, axis=0)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(num_cls, name='output_layer')(x)

    return y_pred



def build_classifier(astromer, params, astromer_trainable, num_cls=None, arch='avg_mlp'):
    # Build classifier
    if params['arch'] == 'redux':
        inp_placeholder = build_input_redux(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = encoder(inp_placeholder)

    if params['arch'] == 'zero':
        inp_placeholder = build_input_zero(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = encoder(inp_placeholder, z_by_layer=True)

    if params['arch'] == 'base' or params['arch'] == 'normal':
        inp_placeholder = build_input_base(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = encoder(inp_placeholder, z_by_layer=True)

    if params['arch'] == 'skip':
        inp_placeholder = build_input_skip(params['window_size'])
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = encoder(inp_placeholder)
        
    if params['arch'] == 'nsp':
        inp_placeholder = build_input_nsp(params['window_size'])      
        encoder = astromer.get_layer('encoder')
        encoder.trainable = astromer_trainable
        embedding = encoder(inp_placeholder)
        embedding = tf.slice(embedding, [0, 1, 0], [-1, -1,-1], name='slice_obs')
        embedding = tf.reshape(embedding, [-1, params['window_size'], params['head_dim']*params['num_heads']])

    mask = 1.- inp_placeholder['mask_in']
       
    print('[INFO] Using {} clf architecture with {}'.format(arch, params['arch']))
    if arch == 'mlp_avg':
        output = get_mlp_avg(embedding[-1], mask, num_cls)

    if arch == 'avg_mlp_dp':
        output = get_avg_mlp_dp(embedding[-1], mask, num_cls)
    
    if arch == 'avg_mlp':
        output = get_avg_mlp(embedding[-1], mask, num_cls)

    if arch == 'linear_att':
        output = get_linear(embedding[-1], mask, num_cls)

    if arch == 'skip_avg_mlp':
        output = get_skip_avg_mlp(embedding, mask, num_cls)

    clf = CustomModel(inputs=inp_placeholder, 
                      outputs=output, 
                      name=arch)

    return clf


class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        return {'y_pred': y_pred, 
                'y_true': y}