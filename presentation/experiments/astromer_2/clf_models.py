import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.layers     import Dense
from tensorflow.keras            import Input, Model

def build_mlp_att(astromer, maxlen, n_classes, train_astromer=False):
    serie  = Input(shape=(maxlen, 1), batch_size=None, name='input')
    times  = Input(shape=(maxlen, 1), batch_size=None, name='times')
    mask   = Input(shape=(maxlen, 1), batch_size=None, name='mask')
    print('BUILDING MLP + ATT')
    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer

    mask = 1.-placeholder['mask_in']

    x = encoder(placeholder, training=train_astromer)
    x = x * mask
    x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
    x = Dense(n_classes)(x)
    return Model(inputs=placeholder, outputs=x, name="FCATT")

def get_classifier_by_name(clf_name,
                           config,
                           astromer=None,
                           train_astromer=False):
    num_cls = pd.read_csv(
                os.path.join(config['classification']['data']['path'],
                            'objects.csv')).shape[0]

    if clf_name == 'mlp_att':
        clf_model = build_mlp_att(astromer,
                                  maxlen=config['astromer']['window_size'],
                                  n_classes=num_cls,
                                  train_astromer=config['classification']['train_astromer'])

    return clf_model
