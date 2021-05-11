import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

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
    lc_path = os.path.join(source,'LCs', path)
    observations = pd.read_csv(lc_path)
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


def create_dataset(source='data/raw_data/macho/MACHO/LCs',
                   metadata='data/raw_data/macho/MACHO/MACHO_dataset.dat',
                   target='data/records/macho/',
                   n_jobs=None,
                   subsets_frac=(0.5, 0.25),
                   max_lcs_per_record=100):
    os.makedirs(target, exist_ok=True)

    meta_df = pd.read_csv(metadata)

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

def standardize(tensor, only_magn=False):
    with tf.name_scope("Standardize") as scope:
        if only_magn:
            rest = tf.slice(tensor, [0, 0, 1], [-1, -1, 1])
            tensor = tf.slice(tensor, [0, 0, 0], [-1, -1, 1])

        mean_value = tf.expand_dims(tf.reduce_mean(tensor, 1), 1, name='mean_value')
        std_value = tf.expand_dims(tf.math.reduce_std(tensor, 1), 1, name='std_value')

        normed = tf.where(std_value == 0.,
                         (tensor - mean_value),
                         (tensor - mean_value)/std_value)
        if only_magn:
            normed = tf.concat([normed, rest], 2)

        return normed

def normalize(tensor, only_time=False, min_value=None, max_value=None):
    with tf.name_scope("Normalize") as scope:
        if only_time:
            rest = tf.slice(tensor, [0, 0, 1], [-1, -1, 2])
            tensor = tf.slice(tensor, [0, 0, 0], [-1, -1, 1])

        if min_value is None or max_value is None:
            min_value = tf.expand_dims(tf.reduce_min(tensor, 1), 1, name='min_value')
            max_value = tf.expand_dims(tf.reduce_max(tensor, 1), 1, name='max_value')

        den = (max_value - min_value)
        normed = tf.where(den== 0.,
                         (tensor - min_value),
                         (tensor - min_value)/den)

        if only_time:
            normed = tf.concat([normed, rest], 2)
        return normed

def get_delta(tensor, name='TensorDelta'):
    times0 = tf.concat([[tensor[0]] , tensor[:-1]], axis=0,
             name=name)
    dt = tensor - times0
    return dt

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

def parse_2(sample, input_size, num_cls=None):
    param_dim = tf.cast(tf.shape(sample['input'])[-1], tf.int32)

    single_len = tf.cast(tf.math.floordiv(input_size, 2), tf.int32)

    steps_1 = steps_2 = 0
    if tf.math.less_equal(sample['length'], single_len):
        # masking needed
        serie_1 = sample['input'][0]
        filler  = tf.zeros([single_len - sample['length'], param_dim])
        serie_1 = tf.concat([serie_1, filler], axis=0)
        serie_2 = tf.zeros_like(serie_1)
        steps_1 = sample['length']

    elif tf.math.greater(sample['length'], input_size):
        # no masking needed
        pivot = tf.random.uniform(shape=[1],
                                  minval=0,
                                  maxval=int(sample['length']-input_size),
                                  dtype=tf.dtypes.int32)[0]

        serie_1 = sample['input'][:, pivot:(pivot+single_len), :][0]
        serie_2 = sample['input'][:, (pivot+single_len):(pivot+2*single_len), :][0]
        steps_1 = steps_2 = single_len
    else:
        # masking needed
        serie_1 = sample['input'][:, :single_len , :][0]
        serie_2 = sample['input'][:, single_len: , :][0]
        step_s2 = tf.cast(tf.shape(serie_2)[0], tf.int32)
        filler  = tf.zeros([single_len - step_s2, param_dim])
        serie_2 = tf.concat([serie_2, filler], axis=0)
        steps_1 = single_len
        steps_2 = step_s2

    # input dictionary
    inp_dict = dict()
    inp_dict['serie_1'] = serie_1
    inp_dict['serie_2'] = serie_2

    inp_dict['steps_2'] = steps_2
    inp_dict['steps_1'] = steps_1

    inp_dict['label'] = sample['label']
    inp_dict['lcid']  = sample['lcid']
    inp_dict['num_cls']  = num_cls if num_cls is not None else 2

    return inp_dict

def adjust_length(func, input_len, num_cls):
    def wrap(*args, **kwargs):
        result = func(*args, input_len, num_cls)
        return result
    return wrap

def load_records(source, batch_size, repeat=1, input_len=100, balanced=False,
                finetuning=False):
    num_cls = len(os.listdir(source)) if finetuning else None
    parse_2_adjusted = adjust_length(parse_2, input_len, num_cls)

    if balanced:
        datasets = [tf.data.TFRecordDataset(os.path.join(source, folder, x)) \
                    for folder in os.listdir(source) for x in os.listdir(os.path.join(source, folder))]

        op = lambda x: tf.data.Dataset.from_tensors(x).cache().repeat(repeat).map(_parse).map(parse_2_adjusted)
        datasets = [dataset.interleave(op,
                                       cycle_length = 25,
                                       block_length=1,
                                       deterministic=False,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                    for dataset in datasets]

        datasets = [dataset.shuffle(1000, reshuffle_each_iteration=True) for dataset in datasets]
        dataset  = tf.data.experimental.sample_from_datasets(datasets)
    else:
        datasets = [os.path.join(source, folder, x) for folder in os.listdir(source) \
                    for x in os.listdir(os.path.join(source, folder))]

        dataset = tf.data.TFRecordDataset(datasets)
        dataset = dataset.cache()
        dataset = dataset.map(_parse).map(parse_2_adjusted)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
