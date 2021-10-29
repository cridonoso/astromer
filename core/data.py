import multiprocessing as mp
import dask.dataframe as dd
import tensorflow as tf
import pandas as pd
import numpy as np
import h5py
import json
import logging
import os

from core.masking import get_masked, set_random, get_padding_mask
from joblib import wrap_non_picklable_objects
from joblib import Parallel, delayed
from core.utils import standardize, normalize


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def get_example(lcid, label, lightcurve):
    """
    Create a record example from numpy values.

    Args:
        lcid (string): object id
        label (int): class code
        lightcurve (numpy array): time, magnitudes and observational error

    Returns:
        tensorflow record
    """

    f = dict()

    dict_features={
    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(lcid).encode()])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[lightcurve.shape[0]])),
    }
    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = {}
    for col in range(lightcurve.shape[1]):
        seqfeat = _float_feature(lightcurve[:, col])
        seqfeat = tf.train.FeatureList(feature = [seqfeat])
        dict_sequence['dim_{}'.format(col)] = seqfeat

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                                  feature_lists= element_lists)
    return ex

@wrap_non_picklable_objects
def clear_sample(observations, band=1):
    observations = observations[['mjd', 'mag', 'std', 'band']] # Sanity check
    observations = observations[observations['band']==band]
    observations = observations.dropna()
    observations = observations.sort_values('mjd')
    observations = observations.drop_duplicates(keep='last')
    return observations.values

def write_record(chunk, index, dest, label):
    with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(index)) as writer:
        for oid, sample in zip(chunk.index, chunk):
            ex = get_example(oid, label, sample[:, :-1])
            writer.write(ex.SerializeToString())

def create_records(observations, metadata, dest='.', class_code=None, max_lc_per_record=20000, njobs=1):

    if class_code==None:
        class_code = dict()
        for index, cls_name in enumerate(metadata['class'].unique()):
            class_code[cls_name] = index

    for label, group in metadata.groupby('class'):

        obj_cls = observations[observations['oid'].isin(group['oid'])]

        res = obj_cls.groupby('oid').apply(clear_sample, band=1, meta=('x', 'f8')).compute()

        chunks = np.array_split(res,
                                np.arange(max_lc_per_record,
                                          res.shape[0],
                                          max_lc_per_record))

        cls_dest = os.path.join(dest, label)
        os.makedirs(cls_dest, exist_ok=True)

        var = Parallel(n_jobs=njobs, backend='multiprocessing')(delayed(write_record)(chunk, k, cls_dest, class_code[label]) \
                                    for k, chunk in enumerate(chunks))

    print('[INFO] Records succefully created. Have a good training')

def get_sample(sample, ndims=3):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(ndims):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(ndims):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]

    # errs = tf.slice(sequence, [0, 2], [-1, 1])
    # cond = errs < 1
    input_dict['input'] = sequence#sequence[cond[...,0]]

    return input_dict

def get_first_k_obs(sequence, max_obs):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    serie_len = tf.shape(sequence)[0]
    curr_max_obs = tf.minimum(serie_len, max_obs)
    sequence = tf.slice(sequence, [0,0], [curr_max_obs, -1])
    return sequence, curr_max_obs

def sample_lc(sequence, max_obs):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    serie_len = tf.shape(sequence)[0]
    curr_max_obs = tf.minimum(serie_len, max_obs)

    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

        sequence = tf.slice(sequence, [pivot,0], [curr_max_obs, -1])
    else:
        sequence = tf.slice(sequence, [0,0], [curr_max_obs, -1])

    return sequence, curr_max_obs

def _parse_pt(sample, msk_prob, rnd_prob, same_prob, max_obs, is_train=False):
    '''
    Pretraining formater
    '''

    input_dict = get_sample(sample)

    if is_train:
        sequence, curr_max_obs = sample_lc(input_dict['input'], max_obs)
    else:
        sequence, curr_max_obs = get_first_k_obs(input_dict['input'], max_obs)

    sequence, mean = standardize(sequence, return_mean=True)

    seq_time = tf.slice(sequence, [0, 0], [curr_max_obs, 1])
    seq_magn = tf.slice(sequence, [0, 1], [curr_max_obs, 1])
    seq_errs = tf.slice(sequence, [0, 2], [curr_max_obs, 1])


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

    if curr_max_obs < max_obs:
        filler    = tf.ones([max_obs-curr_max_obs, 1])
        mask_in   = tf.concat([mask_in, filler], 0)
        seq_magn  = tf.concat([seq_magn, 1.-filler], 0)
        seq_time  = tf.concat([seq_time, 1.-filler], 0)
        mask_out  = tf.concat([mask_out, 1.-filler], 0)
        orig_magn = tf.concat([orig_magn, 1.-filler], 0)

    input_dict['output']   = orig_magn
    input_dict['input']    = seq_magn
    input_dict['times']    = seq_time
    input_dict['mask_out'] = mask_out
    input_dict['mask_in']  = mask_in
    input_dict['length']   = time_steps
    input_dict['mean']     = mean
    input_dict['obserr']   = seq_errs

    return input_dict

def adjust_fn(func, *arguments):
    def wrap(*args, **kwargs):
        result = func(*args, *arguments)
        return result
    return wrap

def datasets_by_cls(source):
    objects  = pd.read_csv(source+'_objs.csv')

    datasets = []
    for folder in os.listdir(source):
        cls_chunks = []
        for file in os.listdir(os.path.join(source, folder)):
            cls_chunks.append(os.path.join(source, folder, file))
        ds = tf.data.TFRecordDataset(cls_chunks)
        datasets.append(ds)

    return datasets

def load_records(source, batch_size, max_obs=100,
                msk_frac=0.2, rnd_frac=0.1,
                same_frac=0.1, repeat=1,
                is_train=False):
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
    fn = adjust_fn(_parse_pt, msk_frac, rnd_frac, same_frac, max_obs, is_train)

    if not is_train:
        print('Testing mode')
        chunks = [os.path.join(source, folder, file) \
                    for folder in os.listdir(source) \
                        for file in os.listdir(os.path.join(source, folder))]

        dataset = tf.data.TFRecordDataset(chunks)
        dataset = dataset.map(fn)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset
    else:
        print('Training Mode')
        datasets = datasets_by_cls(source)
        dataset = tf.data.experimental.sample_from_datasets(datasets)
        dataset = dataset.repeat(repeat)
        dataset = dataset.map(fn)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

def formatter(sample, is_train, max_obs, num_cls, norm='zscore'):
    input_dict = get_sample(sample)

    if is_train:
        sequence, curr_max_obs = sample_lc(input_dict['input'], max_obs)
    else:
        sequence, curr_max_obs = get_first_k_obs(input_dict['input'], max_obs)

    if norm == 'min-max':
        sequence = normalize(sequence, return_mean=True)
    else:
        sequence, _ = standardize(sequence, return_mean=True)

    seq_time = tf.slice(sequence, [0, 0], [curr_max_obs, 1])
    seq_magn = tf.slice(sequence, [0, 1], [curr_max_obs, 1])
    seq_errs = tf.slice(sequence, [0, 2], [curr_max_obs, 1])
    mask_in  = tf.zeros([curr_max_obs, 1])

    if curr_max_obs < max_obs:
        filler    = tf.ones([max_obs-curr_max_obs, 1])
        mask_in   = tf.concat([mask_in, filler], 0)
        seq_magn  = tf.concat([seq_magn, 1.-filler], 0)
        seq_time  = tf.concat([seq_time, 1.-filler], 0)

    dictonary = {
            'input': seq_magn,
            'times': seq_time,
            'mask_in': mask_in
            }
    return dictonary, tf.one_hot(input_dict['label'], num_cls)

def load_records_v3(source, batch_size, max_obs=100, repeat=1, is_train=False,
                   num_cls=5, norm='zscore'):
    """
    Specific Task data loader.
    Args:
        source (string): Record folder
        batch_size (int): Batch size
        max_obs (int): Max. number of observation per serie
    Returns:
        Tensorflow Dataset: Iterator withg preprocessed batches
    """
    fn = adjust_fn(formatter, is_train, max_obs, num_cls, norm)
    if is_train:
        print('Testing mode')
        chunks = [os.path.join(source, folder, file) \
                    for folder in os.listdir(source) \
                        for file in os.listdir(os.path.join(source, folder))]

        dataset = tf.data.TFRecordDataset(chunks)
        dataset = dataset.map(fn)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).cache()
        dataset = dataset.prefetch(1)
        return dataset
    else:
        print('Testing mode')
        chunks = [os.path.join(source, folder, file) \
                    for folder in os.listdir(source) \
                        for file in os.listdir(os.path.join(source, folder))]

        dataset = tf.data.TFRecordDataset(chunks)
        dataset = dataset.map(fn)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset
    # else:
    #     print('Training Mode')
    #     datasets = datasets_by_cls(source)
    #     dataset = tf.data.experimental.sample_from_datasets(datasets)
    #     dataset = dataset.repeat(repeat)
    #     dataset = dataset.map(fn)
    #     dataset = dataset.padded_batch(batch_size)
    #     dataset = dataset.prefetch(1)
    #     return dataset.cache()

class generator:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            for x, l, y in zip(hf['embs'], hf['lengths'], hf['labels']):
                yield x, l, y

def create_mask(x, l, y):
    x = tf.slice(x, [0, 0], [l, -1])
    return {'x':x, 'mask':tf.cast(tf.ones(l), tf.bool)}, y

def get_average(inputs, y):
    inputs['x'] = tf.reduce_mean(inputs['x'], 0)
    return inputs, y

def load_embeddings(path, n_classes, batch_size=16, is_train=False, time_avg=False):
    files = [os.path.join(path, x) for x in os.listdir(path)]
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
        generator(n_classes),
        (tf.float32, tf.int32, tf.int32),
        (tf.TensorShape([200, 256]), tf.TensorShape([]), tf.TensorShape([])),
        args=(filename,)))

    ds = ds.map(create_mask)
    if is_train:
        ds = ds.shuffle(1000)

    if time_avg:
        ds = ds.map(get_average)

    ds = ds.padded_batch(batch_size)
    ds = ds.prefetch(1)
    return ds
