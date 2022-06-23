import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from core.preprocess.masking import get_masked, set_random, pad_sequence
from core.utils import standardize
from core.preprocess.tools import adjust_fn
from core.preprocess.records import deserialize
from tqdm import tqdm
from time import time

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def sample_lc(sample, max_obs):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    input_dict = deserialize(sample)
    sequence = input_dict['input']
    serie_len = tf.shape(sequence)[0]
    pivot = 0

    def fn_true():
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-max_obs+1,
                                  dtype=tf.int32)

        return tf.slice(sequence, [pivot,0], [max_obs, -1])


    def fn_false():
        return tf.slice(sequence, [0,0], [serie_len, -1])

    sequence = tf.cond(
                    tf.greater(serie_len, max_obs),
                    true_fn=lambda: fn_true(),
                    false_fn=lambda: fn_false()
                )

    return sequence, input_dict['label'], input_dict['lcid']

def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length-max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced

def get_windows(sample, max_obs):
    input_dict = deserialize(sample)
    sequence = input_dict['input']
    rest = input_dict['length']%max_obs

    pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs),  pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))
    # aqui falta retornar labels y oids
    y = tf.tile([input_dict['label']], [len(splits)])
    ids = tf.tile([input_dict['lcid']], [len(splits)])

    return splits, y, ids


def mask_sample(x, y , i, msk_prob, rnd_prob, same_prob, max_obs):
    '''
    Pretraining formater
    '''
    x = standardize(x, return_mean=False)

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    orig_magn = seq_magn

    # [MASK] values
    mask_out = get_masked(seq_magn, msk_prob)

    # [MASK] -> Same values
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_prob,
                                   name='set_same')

    # [MASK] -> Random value
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_prob,
                                   name='set_random')

    time_steps = tf.shape(seq_magn)[0]

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out, 1-mask_fill], 0)
        mask_in   = tf.concat([mask_in, mask_fill], 0)
        seq_magn   = tf.concat([seq_magn, 1-mask_fill], 0)
        seq_time   = tf.concat([seq_time, 1-mask_fill], 0)
        orig_magn   = tf.concat([orig_magn, 1-mask_fill], 0)

    input_dict = dict()
    input_dict['output']   = orig_magn
    input_dict['input']    = seq_magn
    input_dict['times']    = seq_time
    input_dict['mask_out'] = mask_out
    input_dict['mask_in']  = mask_in
    input_dict['length']   = time_steps
    input_dict['label']    = y
    input_dict['id']       = i

    return input_dict

def format_pretraining(input_dict, nsp=False):
    x = {
    'input':input_dict['input'],
    'times':input_dict['times'],
    'mask_in':input_dict['mask_in']
    }

    y = (input_dict['output'],
         input_dict['label'],
         input_dict['mask_out'])

    return x, y

def format_inference(input_dict, num_cls, get_ids=False):
    x = {
    'input':input_dict['input'],
    'times':input_dict['times'],
    'mask_in':input_dict['mask_in']
    }

    y = tf.one_hot(input_dict['label'], num_cls)
    if get_ids:
        y = (y, input_dict['id'])
    return x, y

def pretraining_pipeline(source, batch_size, max_obs=100, msk_frac=0.2,
                        rnd_frac=0.1, same_frac=0.1, sampling=False,
                        shuffle=False, n_classes=-1):
    """
    Pretraining data loader.
    This method build the ASTROMER input format.
    ASTROMER format is based on the BERT masking strategy.
    Args:
        source (string): Record folder
        batch_size (int): Batch size
        no_shuffle (bool): Do not shuffle training and validation dataset
        max_obs (int): Max. number of observation per serie
        msk_frac (float): fraction of values to be predicted ([MASK])
        rnd_frac (float): fraction of [MASKED] values to replace with random values
        same_frac (float): fraction of [MASKED] values to replace with true values
    Returns:
        Tensorflow Dataset: Iterator withg preprocessed batches
    """
    rec_paths = []
    for folder in os.listdir(source):
        if folder.endswith('.csv'):
            continue
        for x in os.listdir(os.path.join(source, folder)):
            rec_paths.append(os.path.join(source, folder, x))

    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs)
    else:
        fn_0 = adjust_fn(get_windows, max_obs)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    dataset = tf.data.TFRecordDataset(rec_paths)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)

    if not sampling:
        dataset = dataset.flat_map(lambda x,y,i: tf.data.Dataset.from_tensor_slices((x,y,i)))

    dataset = dataset.map(fn_1)

    if n_classes!=-1:
        print('[INFO] Processing labels')
        fn_2 = adjust_fn(format_label, n_classes)
        dataset = dataset.map(fn_2)
    else:
        dataset = dataset.map(format_pretraining)

    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def inference_pipeline(dataset, batch_size, max_obs=200, n_classes=1,
                       shuffle=False, drop_remainder=False, get_ids=False):
    return dataset

def create_generator(list_of_arrays, labels=None, ids=None):

    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k in zip(list_of_arrays, labels, ids):
        yield {'input': i,
               'label':int(j),
               'lcid':str(k),
               'length':int(i.shape[0])}

def load_numpy(samples, ids=None, labels=None, shuffle=False, repeat=1):
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples,labels,ids),
                                         output_types= {'input':tf.float32,
                                                        'label':tf.int32,
                                                        'lcid':tf.string,
                                                        'length':tf.int32},
                                         output_shapes={'input':(None,3),
                                                        'label':(),
                                                        'lcid':(),
                                                        'length':()})
    if shuffle:
        print('[INFO] Shuffling')
        dataset = dataset.shuffle(10000)

    dataset = dataset.repeat(repeat)
    return dataset
