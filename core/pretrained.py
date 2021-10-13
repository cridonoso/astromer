import tensorflow as tf
import pandas as pd
import numpy as np
import json
import time
import os

from core.astromer import get_ASTROMER
from core.data  import load_records
from itertools import zip_longest

# class ASTROMER(object):
#     """
#     ASTROMER Base Clase
#     """
#     def __init__(self, model=None, model_config=None):
#         super(ASTROMER, self).__init__()
#         self.model = model
#         self.model_config = model_config
#
#     def from_pretrained(self, weights):
#         raise NotImplementedError("Please Implement this method")
#
#     def pad_vectors(self, times, magnitudes):
#         """
#         Pad times and magnitudes vectors to be divisible by
#         the encoder dimensionality
#
#         Args:
#             times (list): a list of times-arrays
#             magnitudes (list): a list of magns-arrays
#
#         Returns:
#             A list of times and magnitudes pieces of [encoder-dim] steps.
#             It also returns a list of indices for joining lightcurves segments
#         """
#
#         steps   = self.model.input['input'].shape[1]
#         lengths = [len(x) for x in magnitudes]
#         maxlen  =  max(lengths)
#         rest    = maxlen%steps
#         master  = [0.]*(maxlen + (steps - rest))
#         times_  = times+[master]
#         magns_  = magnitudes+[master]
#
#
#         masks = [np.ones(len(x)) for x in times]
#         masks_  = masks+[master]
#
#         masks = np.array(list(zip_longest(*masks_, fillvalue=0)),
#                               dtype=np.float32).T
#
#         times = np.array(list(zip_longest(*times_, fillvalue=0)),
#                               dtype=np.float32).T
#         magns = np.array(list(zip_longest(*magns_, fillvalue=0)),
#                               dtype=np.float32).T
#
#         n_id = times.shape[1]//steps
#         indices = np.arange(len(times))
#         new_ind = np.tile(indices, [n_id, 1]).flatten()
#         indices = np.argsort(new_ind)
#         new_ind = new_ind[indices]
#
#         #Split time
#         times = np.vstack(np.split(times, n_id, axis=1))
#         times = times[indices]
#         #Split magn
#         magns = np.vstack(np.split(magns, n_id, axis=1))
#         magns = magns[indices]
#         #Split mask
#         masks = np.vstack(np.split(masks, n_id, axis=1))
#         masks = masks[indices]
#
#         valid = [i for i, x in enumerate(magns) if np.sum(x)!=0]
#         times = times[valid][...,None]
#         magns = magns[valid][...,None]
#         masks = masks[valid][...,None]
#         new_ind = new_ind[valid]
#         return times, magns, new_ind, masks
#
#     def encode(self, magnitudes, times=None, batch_size=100):
#         """
#         Transform a lighcurve to its corresponding attention vector.
#
#         Args:
#             magnitudes (list): A list of numpy matrices of dimension [steps, params]
#
#         Returns:
#             numpy: Attention vector
#
#         """
#
#         assert self.model is not None, 'No model was loaded'
#         encoder = self.model.get_layer('encoder')
#
#         if isinstance(magnitudes, tf.data.Dataset):
#             att_vectors = []
#
#             for batch in magnitudes:
#                 att = encoder(batch)
#                 att_vectors.append(att)
#             v = tf.concat(att_vectors, 0).numpy()
#             return v
#
#         lengths = [len(x) for x in times]
#         if isinstance(magnitudes, list):
#             times, magns, indices, masks = self.pad_vectors(times, magnitudes)
#             sequence = tf.concat([times, magns], 2)
#
#         dataset = attention_loader(sequence, masks, batch_size=batch_size)
#
#         att_vectors = []
#         for batch in dataset:
#             att = encoder(batch)
#             att_vectors.append(att)
#
#         v = tf.concat(att_vectors, 0).numpy()
#         df = pd.DataFrame()
#         for i in range(v.shape[1]):
#             df[i] = list(v[:, i, :])
#
#         df['index'] = indices
#         att_final = []
#         for l, (_, group) in zip(lengths, df.groupby('index')):
#             x = np.concatenate(group.iloc[:, :-1].values, 0)
#             att_final.append(np.stack(x[:l]))
#
#         return att_final

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

    def encode_from_records(self, records_dir, batch_size, dest='.', val_data=0):
        objects = pd.read_csv(records_dir+'_objs.csv')
        astromer_size = self.conf['max_obs']
        maxobs = objects['nobs'].max()
        rest = maxobs%astromer_size
        maxobs = maxobs + astromer_size-rest
        n_windows = maxobs//astromer_size

        batches, val_data = load_records(records_dir,
                                         batch_size,
                                         val_data=20, # either fraction (0, 1) or number of samples per class
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

            tf.data.experimental.save(dataset, dest+'/batch_{}'.format(index))

            end = time.time()
            print('{:.2f}'.format(end-start))

def save_emb(x, path):
    df = pd.DataFrame(x.numpy())
    df.to_csv(path, index=False)
