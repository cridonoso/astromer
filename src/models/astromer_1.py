import tensorflow as tf
import numpy as np
import toml
import os

from src.losses             import custom_rmse
from src.losses.rmse import pearson_loss
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras        import Model
import tensorflow as tf
from pathlib import Path
import joblib

from src.layers import Encoder, RegLayer
from src.layers.input import AddMSKToken


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

    return {'input':magnitudes,
            'times':times,
            'mask_in':att_mask}

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
                 m_alpha=-0.5,
                 mask_format='Q',
                 use_leak=False):

    placeholder = build_input(window_size)

    msk_placeholder = AddMSKToken(trainable=True, 
                                  window_size=window_size, 
                                  on=['input'], name='msk_token')(placeholder)

    encoder = Encoder(window_size=window_size,
                      num_layers=num_layers,
                      num_heads=num_heads,
                      head_dim=head_dim,
                      mixer_size=mixer_size,
                      dropout=dropout,
                      pe_base=pe_base,
                      pe_dim=pe_dim,
                      pe_c=pe_c,
                      m_alpha=m_alpha,
                      mask_format=mask_format,
                      use_leak=use_leak,
                      name='encoder')

    x = encoder(msk_placeholder)

    x = RegLayer(name='regression')(x)

    return CustomModel(inputs=placeholder, outputs=x, name="ASTROMER-1")

class CustomModel(Model):
    def __init__(self, loss_format='rmse', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_format = loss_format
        
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            rmse = custom_rmse(y_true=y['target'],
                               y_pred=y_pred,
                               mask=y['mask_out'],
                               weights=y['w_error'])
            
            corr = pearson_loss(y['target'], y_pred, x['mask_in'])
            
            r2_value = custom_r2(y_true=y['target'], 
                                 y_pred=y_pred, 
                                 mask=y['mask_out'])
            
            if self.loss_format == 'rmse':
                loss = rmse
                
            if self.loss_format == 'rmse+p':
                loss = rmse + corr
            
            if self.loss_format == 'p':
                loss = corr
                
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss': loss, 'r_square':r2_value, 'rmse':rmse, 'p': corr}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        rmse = custom_rmse(y_true=y['target'],
                           y_pred=y_pred,
                           mask=y['mask_out'],
                           weights=y['w_error'])
        corr = pearson_loss(y['target'], y_pred, x['mask_in'])
        
        r2_value = custom_r2(y_true=y['target'], 
                             y_pred=y_pred, 
                             mask=y['mask_out'])
        
        if self.loss_format == 'rmse':
            loss = rmse

        if self.loss_format == 'rmse+p':
            loss = rmse + corr

        if self.loss_format == 'p':
            loss = corr
            
        return {'loss': loss, 'r_square':r2_value, 'rmse':rmse, 'p': corr}
    
    def predict_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        return {'reconstruction': y_pred, 
                'magnitudes': y['target'],
                'times': x['times'],
                'probed_mask': y['mask_out'],
               'mask_in': x['mask_in']}