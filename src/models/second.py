''''
ASTROMER + Skip connections + Next Segment Prediction
'''
import tensorflow as tf

from src.layers                 import EncoderSKIP, RegLayer
from tensorflow.keras.layers    import Input, Dense
from src.losses                 import custom_rmse, custom_bce
from src.metrics                import custom_r2, custom_acc
from tensorflow.keras           import Model

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
                 maxlen=100,
                 batch_size=None,
                 pe_c=False):
    
    # LAYERS DEFINITION
    placeholder = build_input(maxlen+3) # +3 BECAUSE WE ADD [SEP] AND [CLS] TOKENS
    encoder = EncoderSKIP(num_layers,
                          d_model,
                          num_heads,
                          dff,
                          base=base,
                          dropout=dropout,
                          pe_c=pe_c,
                          name='encoder')
    regressor  = RegLayer(name='regression')
    classifier = Dense(2, name='RegLayer', activation='softmax')

    # FORWARD GRAPH
    x = encoder(placeholder)
    att_tkn = tf.slice(x, [0,1,0], [-1,-1,-1], name='att_tkn')
    cls_tkn = tf.slice(x, [0,0,0], [-1,1,-1], name='cls_tkn')
    x = regressor(att_tkn)
    y = classifier(cls_tkn)
    return CustomModel(inputs=placeholder,
                       outputs=[x, y],
                       name="ASTROMER_NSP")

class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred, y_pred = self(x)
            rmse = custom_rmse(y_true=y['target'],
                               y_pred=x_pred,
                               mask=y['mask_out'])
            r2_value = custom_r2(y['target'], x_pred, y['mask_out'])
            bce = custom_bce(y['nsp_label'], y_pred)
            acc = custom_acc(y['nsp_label'], y_pred)
            loss = rmse + bce

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
            x_pred, y_pred = self(x, training=False)
            rmse = custom_rmse(y_true=y['target'],
                              y_pred=x_pred,
                              mask=y['mask_out'])
            r2_value = custom_r2(y['target'], x_pred, y['mask_out'])
            bce = custom_bce(y['nsp_label'], y_pred)
            acc = custom_acc(y['nsp_label'], y_pred)
            loss = rmse + bce

        return {'loss':loss,
                'rmse': rmse,
                'r_square':r2_value,
                'bce':bce,
                'acc':acc}
