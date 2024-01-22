''''
ASTROMER + Skip connections + Next Segment Prediction
'''
import tensorflow as tf
import numpy as np
import joblib
import toml
import os 


from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tqdm import tqdm

from src.layers  import SkipEncoder, RegLayer
from src.losses  import custom_rmse
from src.metrics import custom_r2


def build_input(window_size, batch_size=None):
    magnitudes  = Input(shape=(window_size, 1),
                  batch_size=batch_size,
                  name='magnitudes')
    times       = Input(shape=(window_size, 1),
                  batch_size=batch_size,
                  name='times')
    att_mask    = Input(shape=(window_size, 1),
                  batch_size=batch_size,
                  name='att_mask') 

    pholder = {'magnitudes':magnitudes,
               'times':times,
               'att_mask':att_mask}

    return pholder

def get_ASTROMER(num_layers=2,
                 num_heads=2,
                 head_dim=64,
                 mixer_size=256,
                 dropout=0.1,
                 pe_base=1000,
                 pe_dim=128,
                 pe_c=1,
                 window_size=100,
                 batch_size=None,
                 mask_format='first'): # first / zero
    
    placeholder = build_input(window_size)

    print('[INFO] NSP Encoder')
    x = SkipEncoder(window_size=window_size,
                      num_layers=num_layers,
                      num_heads=num_heads,
                      head_dim=head_dim,
                      mixer_size=mixer_size,
                      dropout=dropout,
                      pe_base=pe_base,
                      pe_dim=pe_dim,
                      pe_c=pe_c,
                      name='encoder')(placeholder)

    outputs = RegLayer(name='regresor')(x)

    return CustomModel(inputs=placeholder, outputs=outputs, name="ASTROMER_SKIP")

class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            rmse = custom_rmse(y_true=y['magnitudes'],
                               y_pred=y_pred,
                               mask=y['probed_mask'])

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=y_pred, 
                                 mask=y['probed_mask'])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(rmse, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        rmse = custom_rmse(y_true=y['magnitudes'],
                           y_pred=y_pred,
                           mask=y['probed_mask'])

        r2_value = custom_r2(y_true=y['magnitudes'], 
                             y_pred=y_pred, 
                             mask=y['probed_mask'])
        return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}

