''''
ASTROMER + Skip connections + Next Segment Prediction
'''
import tensorflow as tf
import toml
import os 

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tqdm import tqdm
from src.layers  import Encoder, ConcatEncoder, TransformLayer, RegLayer, NSPEncoder
from src.losses  import custom_rmse, custom_bce
from src.metrics import custom_r2, custom_acc


def build_input(window_size):
    magnitudes  = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='magnitudes')
    times       = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='times')
    att_mask    = Input(shape=(window_size, 1),
                  batch_size=None,
                  name='att_mask') 
    seg_emb     = Input(shape=(window_size, 1),
                  batch_size=None,
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
                 average_layers=False,
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
                         average_layers=average_layers,
                         mask_format=mask_format,
                         name='encoder')

    reg_layer = TransformLayer(name='regressor')
    
    x = encoder(placeholder)
    outputs = reg_layer(x)
    return Model(inputs=placeholder, outputs=outputs, name="ASTROMER_NSP")

@tf.function
def train_step(model, x, y, optimizer, rmse_factor=0.5):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        
        rmse = custom_rmse(y_true=y['magnitudes'],
                           y_pred=outputs['reconstruction'],
                           mask=y['probed_mask'])

        bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
        
        loss = rmse_factor*rmse + (1.-rmse_factor)*bce

        r2_value = custom_r2(y_true=y['magnitudes'], 
                             y_pred=outputs['reconstruction'], 
                             mask=y['probed_mask'])

        nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return {'loss':loss,
            'rmse': rmse,
            'r_square':r2_value,
            'bce':bce,
            'acc':nsp_acc}

@tf.function
def test_step(model, x, y, rmse_factor=0.5, return_pred=False):
    outputs = model(x, training=False)
    
    rmse = custom_rmse(y_true=y['magnitudes'],
                       y_pred=outputs['reconstruction'],
                       mask=y['probed_mask'])

    bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
    
    loss = rmse_factor*rmse + (1.-rmse_factor)*bce

    r2_value = custom_r2(y_true=y['magnitudes'], 
                         y_pred=outputs['reconstruction'], 
                         mask=y['probed_mask'])

    nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])
    if return_pred:
        return outputs

    return {'loss':loss,
            'rmse': rmse,
            'r_square':r2_value,
            'bce':bce,
            'acc':nsp_acc}

def predict(model, test_loader):
    n_batches = sum([1 for _, _ in test_loader])
    print('[INFO] Processing {} batches'.format(n_batches))
    y_pred = tf.TensorArray(dtype=tf.float32, size=n_batches)
    y_true = tf.TensorArray(dtype=tf.float32, size=n_batches)
    masks  = tf.TensorArray(dtype=tf.float32, size=n_batches)
    times  = tf.TensorArray(dtype=tf.float32, size=n_batches)
    cls_pred = tf.TensorArray(dtype=tf.float32, size=n_batches)
    cls_true = tf.TensorArray(dtype=tf.float32, size=n_batches)

    tbar = tqdm(test_loader, total=n_batches)
    index = 0
    for x, y in tbar:
        outputs = test_step(model, x, y, return_pred=True)
        y_pred = y_pred.write(index, outputs['reconstruction'])
        y_true = y_true.write(index, y['magnitudes'])
        masks  = masks.write(index, y['probed_mask'])
        times = times.write(index, x['times'])
        cls_pred = cls_pred.write(index, outputs['nsp_label'])
        cls_true = cls_true.write(index, y['nsp_label'])
        index+=1

    times = times.concat()
    times = tf.slice(times, [0, 1, 0], [-1, -1, -1])
    y_pred = tf.concat([times, y_pred.concat()], axis=2)
    y_true = tf.concat([times, y_true.concat()], axis=2)
    return y_pred, y_true, masks.concat(), cls_pred.concat(), cls_true.concat()

def restore_model(model_folder):
    with open(os.path.join(model_folder, 'config.toml'), 'r') as f:
        model_config = toml.load(f)


    astromer = get_ASTROMER(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'],
                            encoder_mode=model_config['encoder_mode'],
                            average_layers=model_config['avg_layers'])

    print('[INFO] LOADING PRETRAINED WEIGHTS')
    astromer.load_weights(os.path.join(model_folder, 'weights', 'weights'))

    return astromer, model_config