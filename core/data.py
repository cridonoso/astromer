import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from core.masking import get_masked, set_random, get_padding_mask
from joblib import Parallel, delayed
from tqdm import tqdm

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example(lcid, label, lightcurve):
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

def divide_training_subset(frame, train, val):
    frame = frame.sample(frac=1)
    n_samples = frame.shape[0]
    n_train = int(n_samples*train)
    n_val = int(n_samples*val//2)

    sub_train = frame.iloc[:n_train]
    sub_val   = frame.iloc[n_train:n_train+n_val]
    sub_test  = frame.iloc[n_train+n_val:]

    return ('train', sub_train), ('val', sub_val), ('test', sub_test)

def process_lc(row, source, lc_index, unique_classes, writer):
    path  = row['Path'].split('/')[-1]
    label = list(unique_classes).index(row['Class'])
    lc_path = os.path.join(source, path)
    observations = pd.read_csv(lc_path)
    observations.columns = ['mjd', 'mag', 'errmag']
    observations = observations.dropna()
    observations.sort_values('mjd')
    observations = observations.drop_duplicates(keep='last')
    numpy_lc = observations.values
    ex = get_example(lc_index, label, numpy_lc)
    writer.write(ex.SerializeToString())

def write_records(frame, dest, max_lcs_per_record, source, unique, n_jobs=None):
    n_jobs = mp.cpu_count() if n_jobs is not None else n_jobs
    # Get frames with fixed number of lightcurves
    collection = [frame.iloc[i:i+max_lcs_per_record] \
                  for i in range(0, frame.shape[0], max_lcs_per_record)]
    # Iterate over subset
    for counter, subframe in enumerate(collection):
        with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(counter)) as writer:
            Parallel(n_jobs=n_jobs)(delayed(process_lc)(row, source, k, unique, writer) \
                                    for k, row in subframe.iterrows())


def create_dataset(meta_df,
                   source='data/raw_data/macho/MACHO/LCs',
                   target='data/records/macho/',
                   n_jobs=None,
                   subsets_frac=(0.5, 0.25),
                   max_lcs_per_record=100):
    os.makedirs(target, exist_ok=True)

    bands = meta_df['Band'].unique()
    if len(bands) > 1:
        b = input('Filters {} were found. Type one to continue'.format(' and'.join(bands)))
        meta_df = meta_df[meta_df['Band'] == b]

    unique, counts = np.unique(meta_df['Class'], return_counts=True)
    info_df = pd.DataFrame()
    info_df['label'] = unique
    info_df['size'] = counts
    info_df.to_csv(os.path.join(target, 'objects.csv'), index=False)

    # Separate by class
    cls_groups = meta_df.groupby('Class')

    for cls_name, cls_meta in tqdm(cls_groups, total=len(cls_groups)):
        subsets = divide_training_subset(cls_meta,
                                         train=subsets_frac[0],
                                         val=subsets_frac[0])

        for subset_name, frame in subsets:
            dest = os.path.join(target, subset_name, cls_name)
            os.makedirs(dest, exist_ok=True)
            write_records(frame, dest, max_lcs_per_record, source, unique, n_jobs)

def standardize(tensor, axis=0):
    mean_value = tf.reduce_mean(tensor, axis, name='mean_value')
    std_value = tf.math.reduce_std(tensor, axis, name='std_value')

    if axis == 1:
        mean_value = tf.expand_dims(mean_value, axis)
        std_value = tf.expand_dims(std_value, axis)

    # normed = tf.math.divide_no_nan(tensor - mean_value,
    #                                std_value)
    normed = tensor - mean_value

    return normed

def normalize(tensor, axis=0):
    min_value = tf.reduce_min(tensor, axis, name='min_value')
    max_value = tf.reduce_max(tensor, axis, name='max_value')

    if len(tf.shape(tensor))>2:
        min_value = tf.expand_dims(min_value, axis)
        max_value = tf.expand_dims(max_value, axis)

    normed = tf.math.divide_no_nan(tensor - min_value,
                                   max_value - min_value)
    return normed

def get_delta(tensor):
    tensor = tensor[1:] - tensor[:-1]
    tensor = tf.concat([tf.expand_dims([0.], 1), tensor], 0)
    return tensor

def _parse_normal(sample, max_obs):
    '''
    Pretraining parser
    '''
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
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
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)

    sequence = tf.stack(casted_inp_parameters, axis=2)[0]

    serie_len = tf.shape(sequence)[0]
    curr_max_obs = tf.minimum(serie_len, max_obs)
    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

        sequence = tf.slice(sequence, [pivot,0], [curr_max_obs, -1])
        input_dict['length'] = curr_max_obs
    else:
        sequence = tf.slice(sequence, [0,0], [curr_max_obs, -1])
        input_dict['length'] = curr_max_obs


    seq_time = tf.slice(sequence, [0, 0], [curr_max_obs, 1])
    seq_magn = tf.slice(sequence, [0, 1], [curr_max_obs, 1])
    # seq_magn = standardize(seq_magn)

    # seq_errs = tf.slice(sequence, [0, 2], [curr_max_obs, 1])

    time_steps = tf.shape(seq_magn)[0]
    mask = get_padding_mask(max_obs, tf.expand_dims(input_dict['length'], 0))

    input_dict['input']  = seq_magn
    input_dict['times']  = seq_time
    input_dict['mask']   = tf.transpose(mask)
    input_dict['length'] = time_steps

    return input_dict

def _parse_pt(sample, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs):
    '''
    Pretraining parser
    '''
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
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
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]

    serie_len = tf.shape(sequence)[0]
    curr_max_obs = tf.minimum(serie_len, max_obs)
    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-curr_max_obs,
                                  dtype=tf.int32)

        sequence = tf.slice(sequence, [pivot,0], [curr_max_obs, -1])
        input_dict['length'] = curr_max_obs
    else:
        sequence = tf.slice(sequence, [0,0], [curr_max_obs, -1])
        input_dict['length'] = curr_max_obs


    seq_time = tf.slice(sequence, [0, 0], [curr_max_obs, 1])
    seq_magn = tf.slice(sequence, [0, 1], [curr_max_obs, 1])
    # seq_magn = standardize(seq_magn)
    # seq_errs = tf.slice(sequence, [0, 2], [curr_max_obs, 1])

    # [MASK] values
    mask = get_masked(seq_magn, msk_prob)

    # [MASK] -> Same values
    seq_magn, mask = set_random(seq_magn,
                                mask,
                                seq_magn,
                                same_prob,
                                name='set_same')

    # [MASK] -> Random value
    seq_magn, mask = set_random(seq_magn,
                                mask,
                                tf.random.shuffle(seq_magn),
                                rnd_prob,
                                name='set_random')

    time_steps = tf.shape(seq_magn)[0]

    input_dict['input']  = seq_magn
    input_dict['times']  = seq_time
    input_dict['mask']   = tf.reshape(mask, [time_steps, 1])
    input_dict['length'] = time_steps

    return input_dict

def adjust_fn(func, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs)
        return result
    return wrap

def adjust_fn_clf(func, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, max_obs)
        return result
    return wrap

def pretraining_records(source, batch_size, repeat=1, max_obs=100, nsp_prob=0.5, msk_frac=0.2, rnd_frac=0.1, same_frac=0.1):

    datasets = [os.path.join(source, folder, x) for folder in os.listdir(source) \
                for x in os.listdir(os.path.join(source, folder))]

    dataset = tf.data.TFRecordDataset(datasets)
    fn = adjust_fn(_parse_pt, nsp_prob,
                   msk_frac, rnd_frac, same_frac, max_obs)

    dataset = dataset.repeat(repeat).map(fn).cache()
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def clf_records(source, batch_size, max_obs=100, repeat=1):

    datasets = [os.path.join(source, folder, x) for folder in os.listdir(source) \
                for x in os.listdir(os.path.join(source, folder))]

    dataset = tf.data.TFRecordDataset(datasets)
    fn = adjust_fn_clf(_parse_normal, max_obs)

    dataset = dataset.repeat(repeat).map(fn).cache()
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
