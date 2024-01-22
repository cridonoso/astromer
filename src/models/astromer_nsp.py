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
from src.layers  import Encoder, TransformLayer, RegLayer, NSPEncoder
from src.losses  import rmse_for_nsp, custom_bce
from src.metrics import custom_r2, custom_acc


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
                 encoder_mode='normal',
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

    reg_layer = TransformLayer(name='regressor')
    x = encoder(placeholder)
    outputs = reg_layer(x)
    return CustomModel(inputs=placeholder, outputs=outputs, name="ASTROMER_GAP")

class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)  # Forward pass

            rmse = rmse_for_nsp(y_true=y['magnitudes'],
                               y_pred=outputs['reconstruction'],
                               mask=y['probed_mask'],
                               nsp_label=y['nsp_label'],
                               segment_emb=y['seg_emb'])

            bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
            
            loss = rmse + bce

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=outputs['reconstruction'], 
                                 mask=y['probed_mask'])

            nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'bce':bce,
                'acc':nsp_acc}

    def test_step(self, data):
        x, y = data
        outputs = self(x, training=False)
        rmse = rmse_for_nsp(y_true=y['magnitudes'],
                           y_pred=outputs['reconstruction'],
                           mask=y['probed_mask'],
                           nsp_label=y['nsp_label'],
                           segment_emb=y['seg_emb'])

        bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
        
        loss = rmse + bce

        r2_value = custom_r2(y_true=y['magnitudes'], 
                             y_pred=outputs['reconstruction'], 
                             mask=y['probed_mask'])

        nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])

        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'bce':bce,
                'acc':nsp_acc}

