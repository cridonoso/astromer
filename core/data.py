import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from joblib import Parallel, delayed


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

def get_example(inp, tar, random_cond, label, label2):
    f = dict()
    f['length'] = _int64_feature(tar.shape[0])
    f['class'] = _int64_feature(random_cond)
    f['label'] = _bytes_feature(label.encode('utf-8'))
    f['label2'] = _bytes_feature(label2.encode('utf-8'))

    f['x_times'] = _float_feature(inp[:, 0].flatten().tolist())
    f['x_magn'] = _float_feature(inp[:, 1].flatten().tolist())
    f['x_std'] = _float_feature(inp[:, 2].flatten().tolist())

    f['y_times'] = _float_feature(tar[:, 0].flatten().tolist())
    f['y_magn'] = _float_feature(tar[:, 1].flatten().tolist())
    f['y_std'] = _float_feature(tar[:, 2].flatten().tolist())
    
    ex = tf.train.Example(features=tf.train.Features(feature=f))
    return ex

def process_lc(path_0, meta_df, root, max_inp_len, 
                max_tar_len, label):
    path_1 = meta_df.sample(n=1)
    label2 = path_1['Class'].values[0]
    path_1 = path_1['Path'].values[0]

    current_lc = pd.read_csv(root+path_0)
    random_lc  = pd.read_csv(root+path_1)

    # Sort values by time
    current_lc = current_lc.sort_values('mjd')
    random_lc = random_lc.sort_values('mjd')

    # Remove Nan values
    current_lc = current_lc.dropna()
    random_lc  = random_lc.dropna()

    # Split lightcurve
    time_steps = current_lc.shape[0]
    start = int(max_inp_len)
    end = int(max_inp_len) + max_tar_len

    ex = None
    if time_steps >= end:
        pre_x = current_lc.iloc[:start, :]
        pos_x = current_lc.iloc[start:end, :]

        random_cond = np.random.randint(2)

        if random_cond:
            pos_x = random_lc.iloc[:max_tar_len, :]       
            
        final_x = pd.concat([pre_x, pos_x])
        
        ex = get_example(pre_x.values, 
                         pos_x.values, 
                         random_cond,
                         label,
                         label2)
    return ex

def create_dataset(max_inp_len=100, 
                   max_tar_len=100, 
                   source='data/raw_data/macho/MACHO/',
                   target='data/records/macho/',
                   n_jobs=None):
            
    n_jobs = mp.cpu_count() if n_jobs is not None else n_jobs
    metadata = source+'MACHO_dataset.dat'
    meta_df = pd.read_csv(metadata)
    
    # Separate by classes
    grp_class = meta_df.groupby('Class')

    # Iterate over lightcurves
    for label, lab_frame in grp_class:
        response = Parallel(n_jobs=n_jobs)(delayed(process_lc)\
                    (path_0, meta_df, source, 
                    max_inp_len, max_tar_len, label) \
                    for path_0 in lab_frame['Path'])

        response = [r for r in response if r is not None]

        # Save TF records
        os.makedirs(target, exist_ok=True)
        with tf.io.TFRecordWriter('{}/{}.record'.format(target, label)) as writer:
            for ex in response:
                writer.write(ex.SerializeToString())

def standardize(tensor):
    mean_value = tf.expand_dims(tf.reduce_mean(tensor, 0), 0,
                name='min_value')
    std_value = tf.expand_dims(tf.math.reduce_std(tensor, 0), 0,
                name='max_value')
    normed = tf.where(std_value == 0.,
                     (tensor - mean_value),
                     (tensor - mean_value)/std_value)
    return normed

def normalice(tensor):
    min_value = tf.expand_dims(tf.reduce_min(tensor, 0), 0,
                name='min_value')
    max_value = tf.expand_dims(tf.reduce_max(tensor, 0), 0,
                name='max_value')
    den = (max_value - min_value)
    normed = tf.where(den== 0.,
                     (tensor - min_value),
                     (tensor - min_value)/den)
    return normed

def _parse(sample):
    feat_keys = dict() # features for record


    for k in ['times', 'magn', 'std']:
        feat_keys["x_{}".format(k)] = tf.io.FixedLenSequenceFeature([],
                                                        dtype=tf.float32,
                                                        allow_missing=True)
        feat_keys["y_{}".format(k)] = tf.io.FixedLenSequenceFeature([],
                                                        dtype=tf.float32,
                                                        allow_missing=True)



    feat_keys['length'] = tf.io.FixedLenFeature([], tf.int64)
    feat_keys['class'] = tf.io.FixedLenFeature([], tf.int64)
    feat_keys['label'] = tf.io.FixedLenFeature([], tf.string)
    feat_keys['label2'] = tf.io.FixedLenFeature([], tf.string)

    ex1 = tf.io.parse_example(sample, feat_keys)

    ex1['x_times'] = normalice(ex1['x_times'])
    ex1['y_times'] = normalice(ex1['y_times'])

    SEPTOKEN = tf.expand_dims(101, 0)
    SEPTOKEN = tf.cast(SEPTOKEN, tf.float32)
    class_id = tf.cast(ex1['class'], tf.float32)
    class_id = tf.expand_dims(class_id, 0)

    first_serie  = tf.stack([ex1['x_times'],
                              ex1['x_magn']], 
                              1)
    second_serie = tf.stack([ex1['y_times'],
                              ex1['y_magn']], 
                              1)

    return first_serie, second_serie, tf.cast(ex1['length'], tf.int32), class_id


def load_records(source, batch_size):

    files = [os.path.join(source,x) for x in os.listdir(source)]  
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda x: _parse(x),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()

    val = dataset.take(1000)
    train = dataset.skip(1000)

    train_batches = train.padded_batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batches = val.padded_batch(batch_size)
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
    val_batches = val_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_batches, val_batches

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

# create_dataset()