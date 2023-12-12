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
from src.layers  import Encoder, ConcatEncoder, TransformLayer, RegLayer, NSPEncoder
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
def train_step(model, x, y, optimizer, **kwargs):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        
        rmse = rmse_for_nsp(y_true=y['magnitudes'],
                           y_pred=outputs['reconstruction'],
                           mask=y['probed_mask'],
                           nsp_label=y['nsp_label'],
                           segment_emb=y['seg_emb'])

        bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
        
        loss = kwargs['rmse_factor']*rmse + (1.-kwargs['rmse_factor'])*bce

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
def test_step(model, x, y, return_pred=False, **kwargs):
    outputs = model(x, training=False)
    
    rmse = rmse_for_nsp(y_true=y['magnitudes'],
                       y_pred=outputs['reconstruction'],
                       mask=y['probed_mask'],
                       nsp_label=y['nsp_label'],
                       segment_emb=y['seg_emb'])

    bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
    
    loss = kwargs['rmse_factor']*rmse + (1.-kwargs['rmse_factor'])*bce

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
    
    y_pred = []
    y_true = []
    masks  = []
    times  = []
    cls_pred = []
    cls_true = []

    tbar = tqdm(test_loader, total=n_batches)
    index = 0
    for x, y in tbar:
        outputs = test_step(model, x, y, return_pred=True, rmse_factor=0.5)
        y_pred.append(outputs['reconstruction'])
        y_true.append(y['magnitudes'])
        
        masks.append(y['probed_mask'])
        times.append(x['times'])
        cls_pred.append(outputs['nsp_label'])
        cls_true.append(y['nsp_label'])
     
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)
    times  = tf.concat(times, axis=0)
    masks  = tf.concat(masks, axis=0)
    cls_pred = tf.concat(cls_pred, axis=0)
    cls_true = tf.concat(cls_true, axis=0)
    y_pred = tf.concat([times, y_pred], axis=2)
    y_true = tf.concat([times, y_true], axis=2)
    return y_pred, y_true, masks, cls_pred, cls_true

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


def get_embeddings(astromer, dataset, model_config):
    encoder = astromer.get_layer('encoder')
    embeddings = []
    for x, y in dataset:
        Z = encoder(x)
        embeddings.append(Z.numpy())
        
    max_seq_len = model_config['window_size']
    embedding_dim = model_config['mixer']

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def save_embeddings(embeddings, output_path, file_name):
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, '{}.joblib'.format(file_name))   
    with open(path, "wb") as f:
        joblib.dump(embeddings, f)
    print(f"[INFO] Successfully stored embeddings at path {path}")

