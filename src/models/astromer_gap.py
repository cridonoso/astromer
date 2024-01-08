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
from src.layers  import Encoder, TransformLayer_GAP, RegLayer, NSPEncoder
from src.losses  import custom_rmse, rmse_for_delta_gap, rmse_for_gap
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
    seg_emb     = Input(shape=(window_size, 1),
                  batch_size=batch_size,
                  name='seg_emb')

    pholder = {'magnitudes':magnitudes,
               'times':times,
               'att_mask':att_mask,
               'seg_emb':seg_emb}

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
    encoder = NSPEncoder(window_size=window_size,
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

    reg_layer = TransformLayer_GAP(name='regressor')
    
    x = encoder(placeholder)
    x_gap, x_rec, x_gap_rec = reg_layer(x)

    outputs = {'gap': x_gap, 'reconstruction':x_rec, 'gap_rec': x_gap_rec}

    return CustomModel(inputs=placeholder, outputs=outputs, name="ASTROMER_GAP")

class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)  # Forward pass
            rmse = custom_rmse(y_true=y['magnitudes'],
                               y_pred=outputs['reconstruction'],
                               mask=y['probed_mask'])

            rmse_dt_gap = rmse_for_delta_gap(y_true=y['gap_dt'], 
                                          y_pred=outputs['gap'])
            
            rmse_gap = rmse_for_gap(y_true=y['magnitudes'],
                                    y_pred=outputs['gap_rec'],
                                    gap_mask=y['gap_mask'])

            loss = rmse + rmse_dt_gap + rmse_gap

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=outputs['reconstruction'], 
                                 mask=y['probed_mask'])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'rmse_dt_gap':rmse_dt_gap,
                'rmse_gap': rmse_gap}

    def test_step(self, data):
        x, y = data
        outputs = self(x, training=False)
        rmse = custom_rmse(y_true=y['magnitudes'],
                           y_pred=outputs['reconstruction'],
                           mask=y['probed_mask'])

        rmse_dt_gap = rmse_for_delta_gap(y_true=y['gap_dt'], 
                                      y_pred=outputs['gap'])

        rmse_gap = rmse_for_gap(y_true=y['magnitudes'],
                                y_pred=outputs['gap_rec'],
                                gap_mask=y['gap_mask'])

        loss = rmse + rmse_dt_gap + rmse_gap

        r2_value = custom_r2(y_true=y['magnitudes'], 
                             y_pred=outputs['reconstruction'], 
                             mask=y['probed_mask'])
        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'rmse_dt_gap':rmse_dt_gap,
                'rmse_gap': rmse_gap}
