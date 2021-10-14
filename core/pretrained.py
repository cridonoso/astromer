import tensorflow as tf
import pandas as pd
import numpy as np
import json
import time
import os

from core.astromer import get_ASTROMER
from core.data  import load_records
from itertools import zip_longest


def batch_inference(x, t, m, encoder):
    inputs = {'input':x,
              'times':t,
              'mask_in':m}
    emb = encoder(inputs)
    return emb.numpy()

class ASTROMER_v1:
    """ ASTROMER pretrained model trained on reconstructions only.
        i.e., without next sentence prediction.
        This is the first version uploaded on September 27 2021.
    """

    def __init__(self, project_dir='./weights/astromer_10022021'):
        conf_file = os.path.join(project_dir, 'conf.json')
        with open(conf_file, 'r') as handle:
            self.conf = json.load(handle)

        self.model = get_ASTROMER(num_layers=self.conf['layers'],
                             d_model   =self.conf['head_dim'],
                             num_heads =self.conf['heads'],
                             dff       =self.conf['dff'],
                             base      =self.conf['base'],
                             dropout   =self.conf['dropout'],
                             maxlen    =self.conf['max_obs'],
                             use_leak  =self.conf['use_leak'])

        weights_path = '{}/weights'.format(project_dir)
        self.model.load_weights(weights_path)

    def encode_from_records(self, records_dir, batch_size, dest='.', val_data=-1):
        objects = pd.read_csv(records_dir+'_objs.csv')
        astromer_size = self.conf['max_obs']
        maxobs = objects['nobs'].max()
        rest = maxobs%astromer_size
        maxobs = maxobs + astromer_size-rest
        n_windows = maxobs//astromer_size

        if val_data < 0:
            val_data = self.conf['valptg']

        batches, val_data = load_records(records_dir,
                                         batch_size,
                                         val_data=val_data, # either fraction (0, 1) or number of samples per class
                                         no_shuffle=False,
                                         max_obs=maxobs,
                                         msk_frac=0.,
                                         rnd_frac=0.,
                                         same_frac=0.,
                                         repeat=1)

        os.makedirs(dest, exist_ok=True)

        encoder = self.model.get_layer('encoder')

        embeddings = []
        for index, batch in enumerate(batches):
            start = time.time()
            w_inp  = tf.split(batch['input'], n_windows, axis=1)
            w_time = tf.split(batch['times'], n_windows, axis=1)
            w_mask = tf.split(batch['mask_in'], n_windows, axis=1)

            batch_windows = []
            for x,t,m in zip(w_inp, w_time, w_mask):
                embs = batch_inference(x,t,m,encoder)
                batch_windows.append(embs)

            batch_emb = tf.concat(batch_windows, 1)
            bool_mask = tf.logical_not(tf.cast(tf.squeeze(batch['mask_in']), tf.bool))
            valid_emb = tf.ragged.boolean_mask(batch_emb, bool_mask)
            valid_t = tf.ragged.boolean_mask(batch['times'], bool_mask)
            valid_x = tf.ragged.boolean_mask(batch['input'], bool_mask)

            dataset = tf.data.Dataset.from_tensor_slices((batch['lcid'],
                                                          batch['label'],
                                                          valid_x,
                                                          valid_t,
                                                          valid_emb))

            tf.data.experimental.save(dataset,
                            dest+'/batch_{}'.format(index),
                            compression='GZIP')

            end = time.time()
            print('{:.2f}'.format(end-start))
