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

            dataset = dataset.map(tf_serialize_example)
            filename = dest+'/batch_{}'.format(index)
            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(dataset)

            end = time.time()
            print('{:.2f}'.format(end-start))

def tf_serialize_example(f0,f1,f2,f3,f5):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3, f5),  # Pass these args to the above function.
        tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature0, feature1, feature2, feature3, feature4):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    dict_features = {
      'oid': _bytes_feature(feature0),
      'label': _int64_feature(feature1),
      'input':_float_feature(feature2),
      'times':_float_feature(feature3)
    }
    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = {}
    for col in range(feature4.shape[1]):
      seqfeat = _float_feature(feature4[:, col])
      seqfeat = tf.train.FeatureList(feature = [seqfeat])
      dict_sequence['emb_{}'.format(col)] = seqfeat

    # Create a Features message using tf.train.Example.
    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                              feature_lists= element_lists)
    return ex.SerializeToString()
