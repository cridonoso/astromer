import tensorflow as tf
import os

from core.data.record import deserialize
from core.data.preprocessing import to_windows, standardize
from core.data.masking import mask_dataset

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

def pretraining_pipeline(dataset,
                         batch_size=None,
                         shuffle=False,
                         repeat=1,
                         cache=False,
                         window_size=200,
                         sampling=True,
                         msk_frac=.5,
                         rnd_frac=.2,
                         same_frac=.2,
                         return_ids=False,
                         return_lengths=False):
    """
    Pretraining pipeline.
    Create an ad-hoc ASTROMER dataset

    Args:
        dataset: tf.Dataset (use load_records or load_numpy first)
        batch_size (integer): Number of windows per batch
        window_size: Maximum window size. window_size<=max.length from lightcurves
        sampling: Windows extraction strategy.
                  If True, windows are randomnly sampled from the light curves
                  If False, lightcurves are divided in sequential windows
                  without overlaping.
        msk_frac: observations fraction per light curve that will be masked
        rnd_frac: fraction from masked values to be replaced by random values
        same_frac: fraction from masked values to be replace by same values
        return_ids: Not necessary when training.
        return_lengths: Not necessary when training.

    Returns:
        type: tf.Dataset
    """
    assert isinstance(dataset, (list, str)), '[ERROR] Invalid format'
    assert batch_size is not None, '[ERROR] Undefined batch size'

    if isinstance(dataset, list):
        dataset = load_numpy(dataset)

    if isinstance(dataset, str):
        dataset = load_records(dataset)

    # REPEAT LIGHT CURVES
    if repeat is not None:
        dataset = dataset.repeat(repeat)

    # CREATE WINDOWS
    dataset = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)

    # CREATE BATCHES
    dataset = dataset.padded_batch(batch_size)

    # NORMALICE WINDOWS
    dataset = dataset.map(standardize,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        SHUFFLE_BUFFER = 100
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    if cache:
        dataset = dataset.cache()


    # MASKING
    dataset = mask_dataset(dataset,
                           msk_frac=msk_frac,
                           rnd_frac=rnd_frac,
                           same_frac=same_frac)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(x,
                                                        return_ids,
                                                        return_lengths),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #PREFETCH BATCHES
    dataset = dataset.prefetch(2)

    return dataset

def format_inp_astromer(batch, return_ids=False, return_lengths=False):
    """
    Buildng ASTROMER input

    Args:
        batch (type): a batch of windows and their properties

    Returns:
        type: A tuple (x, y) tuple where x are the inputs and y the labels
    """

    inputs = {
        'input': batch['input_modified'],
        'times': tf.slice(batch['input'], [0,0,0], [-1,-1,1]),
        'mask_in': batch['mask_in']
    }
    outputs = {
        'target': tf.slice(batch['input'], [0,0,1], [-1,-1,1]),
        'mask_out': batch['mask_out']
    }

    if return_ids:
        inputs['ids'] = batch['ids']
    if return_lengths:
        inputs['length'] = batch['length']

    return inputs, outputs
