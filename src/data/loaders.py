import tensorflow as tf
import multiprocessing
import os

from src.data.record import deserialize
from src.data.preprocessing import to_windows, min_max_scaler, standardize_dataset, standardize
from src.data.masking import get_probed, add_random
from src.data.nsp import randomize, randomize_v2

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


def format_input(input_dict, cls_token=None, num_cls=None, test_mode=False):
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
    
    if test_mode:
        print('[INFO] TESTING MODE')
        inputs['original'] = input_dict['original']
        inputs['mask'] = input_dict['mask']

    if num_cls is not None:
        outputs = tf.one_hot(input_dict['label'], num_cls)

    else:
        outputs = {
            'magnitudes': tf.slice(input_dict['input_pre_nsp'], [0, 0, 1], [-1, -1, 1]),
            'nsp_label': input_dict['nsp_label'],
            'probed_mask': tf.expand_dims(input_dict['probed_mask'], -1),
        }

    return inputs, outputs

def format_input_no_nsp(input_dict, num_cls=None, test_mode=False):
    times = tf.slice(input_dict['input'], [0, 0, 0], [-1, -1, 1])
    magnitudes = tf.slice(input_dict['input'], [0, 0, 1], [-1, -1, 1])
    att_mask = tf.expand_dims(input_dict['att_mask'], axis=-1)

    inputs = {
        'magnitudes': magnitudes,
        'times': times,
        'att_mask': att_mask,
    }
    if test_mode:
        print('[INFO] TESTING MODE')
        inputs['original'] = input_dict['original']
        inputs['mask'] = input_dict['mask']

    if num_cls is not None:
        outputs = tf.one_hot(input_dict['label'], num_cls)

    else:
        outputs = {
            'magnitudes': tf.slice(input_dict['original'], [0, 0, 1], [-1, -1, 1]),
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
              shuffle=False,
              njobs=None,
              num_cls=None,
              test_mode=False,
              off_nsp=False):

    if njobs is None:
        njobs = multiprocessing.cpu_count()//2

    dataset = load_records(dataset)

    # REPEAT
    dataset = dataset.repeat(repeat)

    # CREATE WINDOWS
    dataset, sizes = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)

    # STANDARDIZE
    dataset = dataset.map(standardize)

    # CREATE BATCHES
    dataset = dataset.padded_batch(batch_size, padded_shapes=sizes)

    
    # MASKING
    dataset = dataset.map(lambda x: get_probed(x, probed=probed, njobs=njobs))
    dataset = dataset.map(lambda x: add_random(x, random_frac=random_same, njobs=njobs))

    # NSP
    if off_nsp:
        dataset = dataset.map(lambda x: format_input_no_nsp(x, num_cls=num_cls, test_mode=test_mode))
    else:
        dataset = dataset.map(lambda x: randomize_v2(x, nsp_prob=nsp_prob))
        dataset = dataset.map(lambda x: format_input(x, num_cls=num_cls, test_mode=test_mode))

    if shuffle:
        SHUFFLE_BUFFER = 10000
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # PREFETCH
    dataset = dataset.prefetch(2)

    return dataset

# ========================================================
def format_input_lc(input_dict, num_cls):
    x = {
        'input': input_dict['input'],
        'mask': input_dict['mask']
    }

    y = tf.one_hot(input_dict['label'], num_cls)
    return x, y

def load_light_curves(dataset, 
                      num_cls=1,
                      batch_size=16, 
                      window_size=200, 
                      repeat=1,
                      cache=False, 
                      njobs=None):
    '''
    Load data for downstream tasks.
    LC without normalizing
    '''
    if njobs is None:
        njobs = multiprocessing.cpu_count()//2
        
    dataset = load_records(dataset)

    dataset, sizes = to_windows(dataset,
                         window_size=window_size,
                         sampling=False)

    dataset = dataset.padded_batch(batch_size, padded_shapes=sizes)

    dataset = dataset.prefetch(2)

    dataset = dataset.map(lambda x: format_input_lc(x, num_cls))

    return dataset