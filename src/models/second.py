''''
ASTROMER + Skip connections + Next Segment Prediction
'''
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from src.layers  import Encoder, CondEncoder, ConcatEncoder, TransformLayer, RegLayer
from src.losses  import custom_rmse, custom_bce
from src.metrics import custom_r2, custom_acc


def build_input(window_size, off_nsp):
    if not off_nsp:
        window_size = window_size + 1

    magnitudes  = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='magnitudes')
    times       = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='times')
    att_mask    = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='att_mask') 
    
    pholder = {'magnitudes':magnitudes,
               'times':times,
               'att_mask':att_mask}

    if not off_nsp:
        pholder['seg_emb'] = Input(shape=(window_size+1, 1),
                                          batch_size=None,
                                          name='seg_emb')

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
                 average_layers=False,
                 off_nsp=False):
    
    # LAYERS DEFINITION
    placeholder = build_input(window_size, off_nsp=off_nsp)



    if encoder_mode == 'normal':
        encoder = Encoder(window_size=window_size,
                          num_layers=num_layers,
                          num_heads=num_heads,
                          head_dim=head_dim,
                          mixer_size=mixer_size,
                          dropout=dropout,
                          pe_base=pe_base,
                          pe_dim=pe_dim,
                          pe_c=pe_c,
                          average_layers=average_layers,
                          name='encoder')

    if encoder_mode == 'concat':
        encoder = ConcatEncoder(window_size=window_size,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                mixer_size=mixer_size,
                                dropout=dropout,
                                pe_base=pe_base,
                                pe_dim=pe_dim,
                                pe_c=pe_c,
                                average_layers=average_layers,
                                name='encoder')

    if encoder_mode == 'conditioned':
        encoder = CondEncoder(window_size=window_size,
                              num_layers=num_layers,
                              num_heads=num_heads,
                              head_dim=head_dim,
                              mixer_size=mixer_size,
                              dropout=dropout,
                              pe_base=pe_base,
                              pe_dim=pe_dim,
                              pe_c=pe_c,
                              average_layers=average_layers,
                              name='encoder')

    if off_nsp:
        reg_layer = RegLayer(name='regressor')    
    else:
        reg_layer = TransformLayer(name='regressor')

    x = encoder(placeholder)

    outputs = reg_layer(x)

    if off_nsp:
        return CustomModel(inputs=placeholder,
                           outputs=outputs,
                           name="ASTROMER")
    else:
        return CustomModelNSP(inputs=placeholder,
                           outputs=outputs,
                           name="ASTROMER_NSP")

class CustomModel(tf.keras.Model):
    def compile(self, rmse_factor=1, *args, **kwargs):
        super().compile(*args, **kwargs)

    '''
    Custom functional model
    '''
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = custom_rmse(y_true=y['magnitudes'],
                               y_pred=x_pred,
                               mask=y['probed_mask'])

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=x_pred, 
                                 mask=y['probed_mask'])

        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss':loss,
                'r_square':r2_value}

    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred = self(x, training=False)
            loss = custom_rmse(y_true=y['magnitudes'],
                               y_pred=x_pred,
                               mask=y['probed_mask'])

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=x_pred, 
                                 mask=y['probed_mask'])

        return {'loss':loss,
                'r_square':r2_value}


class CustomModelNSP(tf.keras.Model):
    def compile(self, rmse_factor=1, off_nsp=False, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.rmse_factor = tf.cast(rmse_factor, tf.float32)
        self.bce_factor = 1 - self.rmse_factor
        self.off_nsp = off_nsp
    '''
    Custom functional model
    '''
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_cls, x_pred = self(x, training=True)
            
            rmse = custom_rmse(y_true=y['magnitudes'],
                               y_pred=x_pred,
                               mask=y['probed_mask'])

            bce = custom_bce(y['nsp_label'], x_cls)
            
            loss = rmse*self.rmse_factor + bce*self.bce_factor

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=x_pred, 
                                 mask=y['probed_mask'])

            acc = custom_acc(y['nsp_label'], x_cls)
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'bce':bce,
                'acc':acc}

    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_cls, x_pred = self(x, training=False)
            rmse = custom_rmse(y_true=y['magnitudes'],
                               y_pred=x_pred,
                               mask=y['probed_mask'])
            bce = custom_bce(y['nsp_label'], x_cls)
            loss = rmse*self.rmse_factor + bce*self.bce_factor

            r2_value = custom_r2(y_true=y['magnitudes'], 
                                 y_pred=x_pred, 
                                 mask=y['probed_mask'])

            acc = custom_acc(y['nsp_label'], x_cls)

        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'bce':bce,
                'acc':acc}
