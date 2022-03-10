import tensorflow as tf
import logging
import os, sys

from tensorflow.keras.layers import Input, Dense
from core.components.decoder   import RegLayer
from core.components.encoder   import Encoder
from tensorflow.keras import Model

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')

class ASTROMER(Model):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 num_heads=4,
                 dff=128,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 maxlen=200):

        super(ASTROMER, self).__init__()
        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               base=base,
                               rate=dropout,
                               use_leak=use_leak,
                               name='encoder')

        self.regressor = RegLayer(name='regression')

    def compile(self, loss_rec=None, metric_rec=None, **kwargs):
        super(ASTROMER, self).compile(**kwargs)
        self.loss_rec = loss_rec
        self.metric_rec = metric_rec

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training)
        x = self.regressor(x)
        return x

    def train_step(self, data):
        x, (y, mask) = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_rec(y, y_pred, mask=mask)
            r2 = self.metric_rec(y, y_pred, mask=mask)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss, 'r2':r2}

    def test_step(self, data):
        x, (y, mask) = data
        y_pred = self(x, training=False)
        loss   = self.loss_rec(y, y_pred, mask=mask)
        r2     = self.metric_rec(y, y_pred, mask=mask)
        return {'loss': loss, 'r2':r2}
