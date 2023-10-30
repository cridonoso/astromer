import tensorflow as tf
import multiprocessing
import os

from src.data.record import deserialize
from src.data.preprocessing import to_windows, standardize, min_max_scaler
from src.data.masking import mask_dataset
from src.data.nsp import apply_nsp


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

def format_inp_astromer(batch, 
                        return_ids=False, 
                        return_lengths=False, 
                        num_cls=None, 
                        nsp_test=False,
                        aversion='0'):
    """
    Buildng ASTROMER input

    Args:
        batch (type): a batch of windows and their properties

    Returns:
        type: A tuple (x, y) tuple where x are the inputs and y the labels
    """

    inputs, outputs = {}, {}

    if aversion == '1':
        inputs['magnitudes'] = batch['input_modified']
        inputs['times']      = tf.slice(batch['input'], [0,0,0], [-1,-1,1])
        inputs['att_mask']   = batch['att_mask']

        outputs['magnitudes']  = tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['error']       = tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['probed_mask'] = batch['probed_mask'],

    if aversion == '2':
        inputs['magnitudes'] = batch['nsp_magnitudes']
        inputs['times']      = batch['nsp_times']
        inputs['att_mask']   = batch['att_mask']
        inputs['seg_emb']    = tf.expand_dims(batch['seg_emb'], axis=-1)
        
        outputs['magnitudes']  = batch['target_magnitudes']
        outputs['error']       = tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['probed_mask'] = batch['probed_mask']
        outputs['nsp_label'] = batch['nsp_label']

    if num_cls is not None:
        outputs = tf.one_hot(batch['label'], num_cls)
    if return_ids:
        outputs['ids']    = batch['lcid']
    if return_lengths:
        outputs['length'] = batch['length']




    return inputs, outputs

def get_loader(dataset,
               batch_size=None,
               window_size=200,
               probed_frac=.2,
               random_frac=.1,
               nsp_prob=0.5,
               sampling=True,
               shuffle=False,
               repeat=1,
               num_cls=None,
               normalize='zero-mean', # 'minmax'
               cache=False,
               return_ids=False,
               return_lengths=False,
               aversion='2'):


    assert isinstance(dataset, (list, str)), '[ERROR] Invalid format'
    assert batch_size is not None, '[ERROR] Undefined batch size'

    if isinstance(dataset, list):
        dataset = load_numpy(dataset)

    if isinstance(dataset, str):
        dataset = load_records(dataset)

    if shuffle:
        SHUFFLE_BUFFER = 10000
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # REPEAT LIGHT CURVES
    if repeat is not None:
        print('[INFO] Repeating dataset x{} times'.format(repeat))
        dataset = dataset.repeat(repeat)

    # CREATE WINDOWS
    dataset = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)

    if normalize == 'zero-mean':
        dataset = dataset.map(standardize)

    if normalize == 'minmax':
        dataset = dataset.map(min_max_scaler)

    print('[INFO] Loading PT task: Masking')
    dataset, shapes = mask_dataset(dataset,
                           msk_frac=probed_frac,
                           rnd_frac=random_frac,
                           same_frac=random_frac,
                           window_size=window_size)

    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    if aversion == '2':
        print('[INFO] NSP format activated')
        dataset = apply_nsp(dataset, nsp_prob)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(x,
                                                return_ids=return_ids,
                                                return_lengths=return_lengths,
                                                num_cls=num_cls,
                                                aversion=aversion),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    #PREFETCH BATCHES
    dataset = dataset.prefetch(2)

    return dataset
