import tensorflow as tf
import multiprocessing
import os

from src.data.record import deserialize
from src.data.preprocessing import to_windows, min_max_scaler
from src.data.masking import get_probed
from src.data.nsp import randomize

def load_records(records_dir):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """
    rec_paths = []
    for folder in os.listdir(records_dir):
        if folder.endswith('.csv'):
            continue
        for x in os.listdir(os.path.join(records_dir, folder)):
            rec_paths.append(os.path.join(records_dir, folder, x))

    dataset = tf.data.TFRecordDataset(rec_paths)    
    dataset = dataset.map(deserialize)
    return dataset

def create_generator(list_of_arrays, labels=None, ids=None):
    """
    Create an iterator over a list of numpy-arrays light curves
    Args:
        list_of_arrays (list): list of variable-length numpy arrays.

    Returns:
        type: Iterator of dictonaries
    """

    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k in zip(list_of_arrays, labels, ids):
        yield {'input': i,
               'label':int(j),
               'lcid':str(k),
               'length':int(i.shape[0])}

def load_numpy(samples,
               labels=None,
               ids=None):
    """
    Load light curves in numpy format

    Args:
        samples (list): list of numpy arrays containing vary-lenght light curves
    Returns:
        type: tf.Dataset
    """

    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples,labels,ids),
                                         output_types= {'input':tf.float32,
                                                        'label':tf.int32,
                                                        'lcid':tf.string,
                                                        'length':tf.int32},
                                         output_shapes={'input':(None,3),
                                                        'label':(),
                                                        'lcid':(),
                                                        'length':()})
    return dataset


def format_input(input_dict, cls_token=None):
    times = tf.slice(input_dict['input'], [0, 0, 0], [-1, -1, 1])
    times = min_max_scaler(times)

    magnitudes = tf.slice(input_dict['nsp_input'], [0, 0, 1], [-1, -1, 1])
    att_mask = tf.expand_dims(input_dict['att_mask'], axis=-1)
    seg_emb  = tf.expand_dims(input_dict['seg_emb'], axis=-1)

    if cls_token is not None:
        inp_shape = tf.shape(input_dict['nsp_input'])
        cls_vector = tf.ones([inp_shape[0], 1, 1], dtype=tf.float32)
        magnitudes = tf.concat([cls_vector*cls_token, magnitudes], axis=1)
        times = tf.concat([1.-cls_vector, times], axis=1)
        att_mask = tf.concat([1.-cls_vector, att_mask], axis=1)
        seg_emb = tf.concat([1.-cls_vector, seg_emb], axis=1)


    inputs = {
        'magnitudes': magnitudes,
        'times': times,
        'att_mask': att_mask,
        'seg_emb': seg_emb,
    }
    outputs = {
        'magnitudes': tf.slice(input_dict['nsp_input'], [0, 0, 1], [-1, -1, 1]),
        'nsp_label': input_dict['nsp_label'],
        'probed_mask': tf.expand_dims(input_dict['probed_mask'], -1),
    }

    return inputs, outputs

def load_data(dataset, 
              batch_size=16, 
              probed=0.4, 
              random_same=0.2, 
              window_size=1000, 
              nsp_prob=.5, 
              repeat=1, 
              sampling=False, 
              njobs=None):

    if njobs is None:
        njobs = multiprocessing.cpu_count()//2

    dataset = load_records(dataset)

    # REPEAT
    dataset = dataset.repeat(repeat)

    # CREATE WINDOWS
    dataset, sizes = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)
    # CREATE BATCHES
    dataset = dataset.padded_batch(batch_size, padded_shapes=sizes)
    
    # MASKING
    dataset = dataset.map(lambda x: get_probed(x, probed=probed, njobs=njobs))
    
    # NSP
    dataset = dataset.map(lambda x: randomize(x, nsp_prob=nsp_prob))

    # FORMAT input 
    dataset = dataset.map(lambda x: format_input(x, cls_token=-99.))

    # PREFETCH
    dataset = dataset.prefetch(2)

    return dataset
