import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from core.layers import Encoder, RegLayer
from core.losses    import custom_rmse

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
    length = Input(shape=(length,),
                  batch_size=None,
                  dtype=tf.int32,
                  name='length')

    return {'input':serie,
            'mask_in':mask,
            'times':times,
            'length':length}

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 rate=0.1,
                 use_leak=False,
                 maxlen=100,
                 batch_size=None):

    placeholder = build_input(maxlen)

    encoder = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=base,
                rate=rate,
                use_leak=False,
                name='encoder')

    x = encoder(placeholder)

    x = RegLayer(name='regression')(x)

    return CustomModel(inputs=placeholder,
                       outputs=x,
                       name="ASTROMER")

class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_pred = self(data)
            mse = custom_rmse(y_true=data['output'],
                              y_pred=x_pred,
                              mask=data['mask_out'])
        grads = tape.gradient(mse, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': mse}

    def test_step(self, data):
        with tf.GradientTape() as tape:
            x_pred = model(data)
            x_true = data['output']
            mse = custom_rmse(y_true=x_true,
                              y_pred=x_pred,
                              mask=data['mask_out'])
        return {'loss': mse}
