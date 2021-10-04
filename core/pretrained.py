import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os

from core.data import attention_loader, process_lc3, emb_records
from core.astromer import get_ASTROMER
from itertools import zip_longest

class ASTROMER(object):
    """
    ASTROMER Base Clase
    """
    def __init__(self, model=None, model_config=None):
        super(ASTROMER, self).__init__()
        self.model = model
        self.model_config = model_config

    def from_pretrained(self, weights):
        raise NotImplementedError("Please Implement this method")

    def pad_vectors(self, times, magnitudes):
        """
        Pad times and magnitudes vectors to be divisible by
        the encoder dimensionality

        Args:
            times (list): a list of times-arrays
            magnitudes (list): a list of magns-arrays

        Returns:
            A list of times and magnitudes pieces of [encoder-dim] steps.
            It also returns a list of indices for joining lightcurves segments
        """

        steps   = self.model.input['input'].shape[1]
        lengths = [len(x) for x in magnitudes]
        maxlen  =  max(lengths)
        rest    = maxlen%steps
        master  = [0.]*(maxlen + (steps - rest))
        times_  = times+[master]
        magns_  = magnitudes+[master]


        masks = [np.ones(len(x)) for x in times]
        masks_  = masks+[master]

        masks = np.array(list(zip_longest(*masks_, fillvalue=0)),
                              dtype=np.float32).T

        times = np.array(list(zip_longest(*times_, fillvalue=0)),
                              dtype=np.float32).T
        magns = np.array(list(zip_longest(*magns_, fillvalue=0)),
                              dtype=np.float32).T

        n_id = times.shape[1]//steps
        indices = np.arange(len(times))
        new_ind = np.tile(indices, [n_id, 1]).flatten()
        indices = np.argsort(new_ind)
        new_ind = new_ind[indices]

        #Split time
        times = np.vstack(np.split(times, n_id, axis=1))
        times = times[indices]
        #Split magn
        magns = np.vstack(np.split(magns, n_id, axis=1))
        magns = magns[indices]
        #Split mask
        masks = np.vstack(np.split(masks, n_id, axis=1))
        masks = masks[indices]

        valid = [i for i, x in enumerate(magns) if np.sum(x)!=0]
        times = times[valid][...,None]
        magns = magns[valid][...,None]
        masks = masks[valid][...,None]
        new_ind = new_ind[valid]
        return times, magns, new_ind, masks

    def encode(self, magnitudes, times, batch_size=100):
        """
        Transform a lighcurve to its corresponding attention vector.

        Args:
            input (list): A list of numpy matrices of dimension [steps, params]

        Returns:
            numpy: Attention vector

        """

        assert self.model is not None, 'No model was loaded'
        encoder = self.model.get_layer('encoder')

        lengths = [len(x) for x in times]
        if isinstance(magnitudes, list):
            times, magns, indices, masks = self.pad_vectors(times, magnitudes)
            sequence = tf.concat([times, magns], 2)

        dataset = attention_loader(sequence, masks, batch_size=batch_size)

        att_vectors = []
        for batch in dataset:
            att = encoder(batch)
            att_vectors.append(att)

        v = tf.concat(att_vectors, 0).numpy()
        df = pd.DataFrame()
        for i in range(v.shape[1]):
            df[i] = list(v[:, i, :])

        df['index'] = indices
        att_final = []
        for l, (_, group) in zip(lengths, df.groupby('index')):
            x = np.concatenate(group.iloc[:, :-1].values, 0)
            att_final.append(np.stack(x[:l]))

        return att_final

    def to_record(self, embeddings, labels, dest='record', oids=None):
        directory = '/'.join(dest.split('/')[:-1])
        os.makedirs(directory, exist_ok=True)

        if oids is None:
         oids=list(range(len(embeddings)))

        with tf.io.TFRecordWriter(dest) as writer:
            for index in range(len(oids)):
                process_lc3(oids[index],
                            labels[index],
                            embeddings[index],
                            writer)

    def load_record(self, folder, batch_size=16, max_obs=200, take=-1, average=False):
        dataset = emb_records(folder, batch_size, max_obs, take, average=average)

        return dataset

class ASTROMER_v1(ASTROMER):
    """ ASTROMER pretrained model trained on reconstructions only.
        i.e., without next sentence prediction.
        This is the first version uploaded on September 27 2021.
    """

    def __init__(self, **kwargs):
        project_dir = './weights/astromer_10022021'
        model, conf = self.from_pretrained(project_dir)
        super(ASTROMER_v1, self).__init__(model=model, model_config=conf)

    def from_pretrained(self, project_dir):
        """
        Loads pretrained model weights.

        Args:
            project_dir (string): Path to the project folder where
                                  config and weights can be found.

        Returns:
            ASTROMER object: ASTROMER_v1 class initialized with the
                             pre-trained weights
        """

        conf_file = os.path.join(project_dir, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        model = get_ASTROMER(num_layers=conf['layers'],
                             d_model   =conf['head_dim'],
                             num_heads =conf['heads'],
                             dff       =conf['dff'],
                             base      =conf['base'],
                             dropout   =conf['dropout'],
                             maxlen    =conf['max_obs'],
                             use_leak  =conf['use_leak'])

        weights_path = '{}/weights'.format(project_dir)
        model.load_weights(weights_path)

        return model, conf
