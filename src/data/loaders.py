import tensorflow as tf
import os

from src.data.record import deserialize
from src.data.preprocessing import to_windows, standardize
from src.data.masking import mask_dataset, mask_sample
from src.data.nsp import nsp_dataset

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
                         window_size=200,
                         msk_frac=.5,
                         rnd_frac=.2,
                         same_frac=.2,
                         sampling=True,
                         shuffle=False,
                         repeat=1,
                         num_cls=None,
                         normalize=True,
                         cache=False,
                         return_ids=False,
                         return_lengths=False,
                         nsp_prob=1.,
                         nsp_frac=0.,
                         moving_window=False,
                         nsp_test=False):
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
        moving_window: Random sengement prediction

    Returns:
        type: tf.Dataset
    """
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
        print(repeat)
        dataset = dataset.repeat(repeat)

    # CREATE WINDOWS
    dataset = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)

    if normalize:
        dataset = dataset.map(standardize)

    dataset = mask_dataset(dataset,
                           msk_frac=msk_frac,
                           rnd_frac=rnd_frac,
                           same_frac=same_frac,
                           window_size=window_size)

    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, ],
              'label' :(),
              'input_modified': [None, None],
              'mask_in': [None, None],
              'mask_out': [None, None]}

    if nsp_frac>0. and nsp_prob<=1.:
        print('[INFO] Using NSP')
        print('[INFO] Mov. win: ',moving_window)
        dataset = nsp_dataset(dataset,
                              prob=nsp_prob,
                              frac=nsp_frac,
                              moving_window=moving_window,
                              buffer_shuffle=5000)
        shapes['nsp_label'] = ()
        shapes['mask'] = (None, None)
        shapes['original_input'] = (None, 3)

    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(x,
                                                        return_ids=return_ids,
                                                        return_lengths=return_lengths,
                                                        num_cls=num_cls,
                                                        nsp_test=nsp_test),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    #PREFETCH BATCHES
    dataset = dataset.prefetch(2)

    return dataset

def format_inp_astromer(batch, return_ids=False, return_lengths=False, num_cls=None, nsp_test=False):
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
    if num_cls is not None:
        outputs = tf.one_hot(batch['label'], num_cls)
    else:
        outputs = {
            'target': tf.slice(batch['input'], [0,0,1], [-1,-1,1]),
            'mask_out': batch['mask_out'],
        }

    if 'nsp_label' in batch.keys():
        outputs['nsp_label'] = batch['nsp_label']
        outputs['target'] = tf.slice(outputs['target'], [0,1,0], [-1,-1,-1])
        outputs['mask_out'] = tf.slice(outputs['mask_out'], [0,1,0], [-1,-1,-1])

    if nsp_test:
        inputs['original_input'] = batch['original_input']
    if return_ids:
        inputs['ids'] = batch['ids']
    if return_lengths:
        inputs['length'] = batch['length']

    return inputs, outputs
