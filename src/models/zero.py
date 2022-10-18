import tensorflow as tf

from src.layers             import Encoder, RegLayer
from src.losses             import custom_rmse
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input
from tensorflow.keras        import Model

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

    return {'input':serie,
            'mask_in':mask,
            'times':times}

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 no_train=True, # WARNING
                 maxlen=100,
                 batch_size=None,
                 pe_v2=False):

    placeholder = build_input(maxlen)

    encoder = Encoder(num_layers,
                      d_model,
                      num_heads,
                      dff,
                      base=base,
                      dropout=dropout,
                      use_leak=use_leak,
                      pe_v2=pe_v2,
                      name='encoder')

    if no_train:
        encoder.trainable = False

    x = encoder(placeholder)

    x = RegLayer(name='regression')(x)

    return CustomModel(inputs=placeholder,
                       outputs=x,
                       name="ASTROMER")

class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred = self(x, training=False)
            mse = custom_rmse(y_true=y['target'],
                              y_pred=x_pred,
                              mask=y['mask_out'])
            r2_value = custom_r2(y['target'], x_pred, y['mask_out'])

        grads = tape.gradient(mse, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': mse, 'r_square':r2_value}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred = self(x, training=False)
            mse = custom_rmse(y_true=y['target'],
                              y_pred=x_pred,
                              mask=y['mask_out'])
            r2_value = custom_r2(y['target'], x_pred, y['mask_out'])
        return {'loss': mse, 'r_square':r2_value}
