import tensorflow as tf
import numpy as np
import toml
import os

from src.losses             import custom_rmse
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras        import Model
import tensorflow as tf
from pathlib import Path
import joblib

from src.layers import Encoder, RegLayer

def build_input(length):
    serie  = Input(shape=(length, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(length, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(length, 1),
                  batch_size=None,
                  name='mask')

    return {'magnitudes':serie,
            'att_mask':mask,
            'times':times}

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
                 mask_format='first'):

    placeholder = build_input(window_size)

    encoder = Encoder(window_size=window_size,
                      num_layers=num_layers,
                      num_heads=num_heads,
                      head_dim=head_dim,
                      mixer_size=mixer_size,
                      dropout=dropout,
                      pe_base=pe_base,
                      pe_dim=pe_dim,
                      pe_c=pe_c,
                      mask_format=mask_format,
                      name='encoder')

    x = encoder(placeholder)
    x = RegLayer(name='regression')(x)

    return CustomModel(inputs=placeholder, outputs=x, name="ASTROMER-1")

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