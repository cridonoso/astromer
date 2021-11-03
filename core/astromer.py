import tensorflow as tf
import logging
import os, sys

from core.encoder   import Encoder

from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 no_train=False,
                 maxlen=200,
                 batch_size=None):

    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = Encoder(num_layers,
                      d_model,
                      num_heads,
                      dff,
                      base=base,
                      rate=dropout,
                      use_leak=use_leak,
                      name='encoder')

    if no_train:
        encoder.trainable = False

    x = encoder(placeholder)

    x = Dense(1, name='Regressor')(x)
    x = LayerNormalization(name='norm')(x)

    return Model(inputs=placeholder,
                 outputs=x,
                 name="ASTROMER")
