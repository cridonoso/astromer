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

    if len(tf.shape(tensor))>2:
        mean_value = tf.expand_dims(mean_value, axis)
        std_value = tf.expand_dims(std_value, axis)

    normed = tf.math.divide_no_nan(tensor - mean_value,
                                   std_value)
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

    curr_max_obs = tf.minimum(input_dict['length'], max_obs)

    seq_time = tf.slice(sequence, [0, 0], [curr_max_obs, 1])
    seq_magn = tf.slice(sequence, [0, 1], [curr_max_obs, 1])
    seq_errs = tf.slice(sequence, [0, 2], [curr_max_obs, 1])

    seq1, seq2   = tf.split(seq_magn, 2, axis=0)
    time1, time2 = tf.split(seq_time, 2, axis=0)
    original  = tf.concat([seq1, seq2], 0)
    # [MASK] values
    mask_1 = get_masked(seq1, msk_prob)
    mask_2 = get_masked(seq2, msk_prob)

    # [MASK] -> Random value
    seq1, mask_1 = set_random(seq1, mask_1, seq2, rnd_prob, name='set_random_1')
    seq2, mask_2 = set_random(seq2, mask_2, seq1, rnd_prob, name='set_random_2')

    # [MASK] -> Same values
    seq1, mask_1 = set_random(seq1, mask_1, seq1, same_prob, name='set_same_1')
    seq2, mask_2 = set_random(seq2, mask_2, seq2, same_prob, name='set_same_2')

    # Next Sentence Prediction
    is_random = tf.random.categorical(tf.math.log([[1-nsp_prob, nsp_prob]]), 1)
    is_random = tf.cast(is_random, tf.bool, name='isRandom')

    if is_random:
        noise = tf.random.normal(tf.shape(seq2),
                                 mean=tf.reduce_mean(seq2),
                                 stddev=tf.math.reduce_std(seq2))
        serie = tf.concat([seq1, tf.random.shuffle(seq2) + noise], 0)
    else:
        serie  = tf.concat([seq1, seq2], 0)

    time_steps = tf.shape(serie)[0]
    times = tf.concat([time1, time2], 0)

    mask = tf.concat([mask_1, mask_2], 0)
    mask = tf.reshape(mask, [time_steps, 1])

    input_dict['input']  = serie
    input_dict['original']  = original
    input_dict['times']  = times
    input_dict['mask']   = mask
    input_dict['length'] = time_steps
    input_dict['label']  = tf.squeeze(tf.cast(is_random, tf.int32))

    return input_dict

def adjust_fn(func, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, nsp_prob, msk_prob, rnd_prob, same_prob, max_obs)
        return result
    return wrap

def pretraining_records(source, batch_size, max_obs=100, nsp_prob=0.5, msk_prob=0.2, rnd_prob=0.1, same_prob=0.1):

    datasets = [os.path.join(source, folder, x) for folder in os.listdir(source) \
                for x in os.listdir(os.path.join(source, folder))]

    dataset = tf.data.TFRecordDataset(datasets)
    fn = adjust_fn(_parse_pt, nsp_prob,
                   msk_prob, rnd_prob, same_prob, max_obs)

    dataset = dataset.map(fn).cache()
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
