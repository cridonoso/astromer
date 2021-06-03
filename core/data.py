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

def standardize_mag(tensor):
    with tf.name_scope("Standardize") as scope:
        times = tf.slice(tensor, [0, 0], [-1, 1])
        std = tf.slice(tensor, [0, 2], [-1, 1])
        tensor = tf.slice(tensor, [0, 1], [-1, 1])

        mean_value = tf.reduce_mean(tensor, 0, name='mean_value')
        std_value = tf.math.reduce_std(tensor, 0, name='std_value')

        normed = tf.where(std_value == 0.,
                         (tensor - mean_value),
                         (tensor - mean_value)/std_value)

        normed = tf.concat([times, normed, std], 1)
        return normed

def normalize(tensor, only_time=False, min_value=None, max_value=None):
    with tf.name_scope("Normalize") as scope:
        min_value = tf.expand_dims(tf.reduce_min(tensor, 1), 1,
                                    name='min_value')
        max_value = tf.expand_dims(tf.reduce_max(tensor, 1), 1,
                                    name='max_value')

        den = (max_value - min_value)
        normed = tf.where(den== 0.,
                         (tensor - min_value),
                         (tensor - min_value)/den)
        return normed

def _parse(sample):


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

    input_dict['input'] = tf.stack(casted_inp_parameters, axis=2)
    return input_dict


def pretrain_input(seq_1, seq_2, nsp_prob, msk_frac, rnd_frac, same_frac,
                    max_obs, clstkn=-99, septkn=-98):

    inp_dim = tf.shape(seq_1['input'], name='inp_dim')
    clstkn = tf.tile(tf.cast([[clstkn]], tf.float32), [1, inp_dim[-1]], name='cls_tkn')
    septkn = tf.tile(tf.cast([[septkn]], tf.float32), [1, inp_dim[-1]], name='sep_tkn')
    msktkn = tf.zeros([1], name='msk_tkn')
    half_obs = max_obs//2

    with tf.name_scope('Split'):
        pivot_1 = tf.random.uniform(shape=[1],
                                    minval=0,
                                    maxval=int(seq_1['length']-max_obs),
                                    dtype=tf.dtypes.int32)[0]

        serie_1 = seq_1['input'][:, pivot_1:(pivot_1+half_obs), :][0]
        serie_1 = standardize_mag(serie_1)
        serie_2 = seq_1['input'][:, (pivot_1+half_obs):(pivot_1+max_obs), :][0]
        serie_2 = standardize_mag(serie_2)
        original = tf.concat([clstkn, serie_1, septkn, serie_2, septkn], 0,
                             name='original')
         # just to plot it

        is_random = tf.random.categorical(tf.math.log([[1-nsp_prob, nsp_prob]]), 1)
        is_random = tf.cast(is_random, tf.bool, name='isRandom')
        if is_random:
            pivot_2 = tf.random.uniform(shape=[1],
                                        minval=0,
                                        maxval=int(seq_2['length']-half_obs),
                                        dtype=tf.dtypes.int32)[0]

            serie_2 = seq_2['input'][:, pivot_2:(pivot_2+half_obs), :][0]
            serie_2 = standardize_mag(serie_2)

    mask_1 = get_masked(serie_1, msk_frac)
    mask_2 = get_masked(serie_1, msk_frac)

    serie_1, mask_1 = set_random(serie_1, mask_1, serie_2, rnd_frac, name='set_random')
    serie_2, mask_2 = set_random(serie_2, mask_2, serie_1, rnd_frac, name='set_random')

    serie_1, mask_1 = set_random(serie_1, mask_1, serie_1, same_frac, name='set_same')
    serie_2, mask_2 = set_random(serie_2, mask_2, serie_2, same_frac, name='set_same')


    serie = tf.concat([clstkn, serie_1, septkn, serie_2, septkn], 0)
    times = tf.slice(serie, [0, 0], [-1, 1], name='times')
    input = tf.slice(serie, [0, 1], [-1, 1], name='input')
    mask  = tf.concat([msktkn, mask_1, msktkn, mask_2, msktkn], 0,
                      name='inp_mask')

    input_dic = {
        'input': input,
        'times': times,
        'segsep': half_obs+2, #segment separator position
        'mask' : tf.expand_dims(mask, 1),
        'label': tf.squeeze(tf.cast(is_random, tf.int32))
    }

    return input_dic

def adjust_fn(func, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs)
        return result
    return wrap

def pretraining_records(source, batch_size, max_obs=100, nsp_prob=0.5, msk_prob=0.2, rnd_prob=0.1, same_prob=0.1):

    datasets = [os.path.join(source, folder, x) for folder in os.listdir(source) \
                for x in os.listdir(os.path.join(source, folder))]

    dataset_1 = tf.data.TFRecordDataset(datasets)
    dataset_2 = tf.data.TFRecordDataset(datasets)

    dataset_1 = dataset_1.map(_parse)
    dataset_2 = dataset_2.map(_parse).shuffle(1000)

    dataset = tf.data.Dataset.zip((dataset_1, dataset_2))

    fn = adjust_fn(pretrain_input, nsp_prob,
                   msk_prob, rnd_prob, same_prob, max_obs)

    dataset = dataset.map(fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def clf_input(seq_1, max_obs, clstkn=-99, septkn=-98):

    inp_dim = tf.shape(seq_1['input'], name='inp_dim')
    clstkn = tf.tile(tf.cast([[clstkn]], tf.float32), [1, inp_dim[-1]], name='cls_tkn')
    septkn = tf.tile(tf.cast([[septkn]], tf.float32), [1, inp_dim[-1]], name='sep_tkn')
    msktkn = tf.zeros([1], name='msk_tkn')
    half_obs = max_obs//2
    
    with tf.name_scope('Split'):
        pivot_1 = tf.random.uniform(shape=[1],
                                    minval=0,
                                    maxval=int(seq_1['length']-max_obs+1),
                                    dtype=tf.dtypes.int32)[0]

        serie_1 = seq_1['input'][:, pivot_1:(pivot_1+max_obs+1), :][0]
        serie_1 = standardize_mag(serie_1)

        mask_1 = tf.sequence_mask(tf.shape(serie_1)[0], max_obs+1)
        mask_1 = tf.cast(tf.logical_not(mask_1), tf.float32)

        serie = tf.concat([clstkn, serie_1, septkn], 0, name='input')
        input = tf.slice(serie, [0, 1], [-1, 1], name='input')
        times = tf.slice(serie, [0, 0], [-1, 1], name='times')


        mask  = tf.concat([msktkn, mask_1, msktkn], 0,
                          name='inp_mask')

    input_dic = {
        'input': input,
        'times': times,
        'mask': mask,
        'segsep': half_obs+2, #segment separator position
        'length': tf.shape(serie)[0],
        'label': tf.cast(seq_1['label'], tf.int32)
    }

    return input_dic

def classification_records(source, batch_size, max_obs=100, take=1):
    datasets = [tf.data.TFRecordDataset(os.path.join(source, folder, x)) \
                        for folder in os.listdir(source) if not folder.endswith('.csv')\
                        for x in os.listdir(os.path.join(source, folder))]
    datasets = [dataset.map(_parse) for dataset in datasets]
    datasets = [dataset.map(lambda x: clf_input(x, max_obs)) for dataset in datasets]
    datasets = [dataset.cache() for dataset in datasets]
    datasets = [dataset.shuffle(1000, reshuffle_each_iteration=True) for dataset in datasets]
    dataset = tf.data.experimental.sample_from_datasets(datasets)
    dataset = dataset.padded_batch(batch_size).prefetch(250)
    return dataset
