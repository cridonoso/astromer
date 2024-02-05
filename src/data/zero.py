import tensorflow as tf
import os

from src.data.masking import get_masked, set_random, mask_sample
from src.data.preprocessing import sample_lc, to_windows
from src.data.loaders import load_records
from src.data.record import deserialize

def mask_dataset(dataset,
                 msk_frac=.5,
                 rnd_frac=.2,
                 same_frac=.2,
                 window_size=None):
    """
    Mask samples per batch following BERT strategy

    Args:
        dataset: A batched tf.Dataset
        msk_frac: observations fraction per light curve that will be masked
        rnd_frac: fraction from masked values to be replaced by random values
        same_frac: fraction from masked values to be replace by same values

    Returns:
        type: tf.Dataset
    """
    assert window_size is not None, 'Masking per sample needs window_size to be specified'
    dataset = dataset.map(lambda x: mask_sample(x,
                                                msk_frac=msk_frac,
                                                rnd_frac=rnd_frac,
                                                same_frac=same_frac,
                                                max_obs=window_size),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
                         normalize='zero-mean', # 'minmax'
                         cache=False,
                         return_ids=False,
                         return_lengths=False,
                         key_format='zero'):
    """ Data preprocessing pipeline. It contains Next Segment Prediciton (NSP)

    :param dataset: Records folder or list of numpy array light curves
    :type dataset: string
    :param batch_size: Number of samples per iteration, defaults to None
    :type batch_size: number, optional
    :param window_size: Subsequence of observations to be sampled from each light curve, defaults to 200
    :type window_size: number, optional
    :param msk_frac: [description], defaults to .5
    :type msk_frac: number, optional
    :param rnd_frac: [description], defaults to .2
    :type rnd_frac: number, optional
    :param same_frac: [description], defaults to .2
    :type same_frac: number, optional
    :param sampling: [description], defaults to True
    :type sampling: bool, optional
    :param shuffle: [description], defaults to False
    :type shuffle: bool, optional
    :param repeat: [description], defaults to 1
    :type repeat: number, optional
    :param num_cls: [description], defaults to None
    :type num_cls: [type], optional
    :param normalize: [description], defaults to 'zero-mean'
    :type normalize: str, optional
    :param # 'minmax' cache: [description], defaults to False
    :type # 'minmax' cache: bool, optional
    :param return_ids: [description], defaults to False
    :type return_ids: bool, optional
    :param return_lengths: [description], defaults to False
    :type return_lengths: bool, optional
    :param nsp_prob: [description], defaults to None
    :type nsp_prob: [type], optional
    :param nsp_frac: [description], defaults to None
    :type nsp_frac: [type], optional
    :param moving_window: [description], defaults to False
    :type moving_window: bool, optional
    :param nsp_test: [description], defaults to False
    :type nsp_test: bool, optional
    :returns: [description]
    :rtype: {[type]}
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
    
    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(x,
                                                return_ids=return_ids,
                                                return_lengths=return_lengths,
                                                num_cls=num_cls,
                                                key_format=key_format),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    #PREFETCH BATCHES
    dataset = dataset.prefetch(2)

    return dataset

def format_inp_astromer(batch, return_ids=False, return_lengths=False, num_cls=None, nsp_test=False, 
                        key_format='0'):
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
        'mask_in': tf.expand_dims(batch['mask_in'], -1),
        'mask_out':tf.expand_dims(batch['mask_out'], -1)
        
    }
        
    if num_cls is not None:
        outputs = tf.one_hot(batch['label'], num_cls)
    else:
        outputs = {
            'target': tf.slice(batch['input'], [0,0,1], [-1,-1,1]),
            'error': tf.slice(batch['input'], [0,0,2], [-1,-1,1]),
            'mask_out': batch['mask_out'],
        }

    if 'nsp_label' in batch.keys() and num_cls is None:
        outputs['nsp_label'] = batch['nsp_label']
        outputs['target'] = tf.slice(outputs['target'], [0,1,0], [-1,-1,-1])
        outputs['mask_out'] = tf.slice(outputs['mask_out'], [0,1,0], [-1,-1,-1])


    if nsp_test:
        inputs['original_input'] = batch['original_input']
        
    if return_lengths:
        inputs['length'] = batch['length']

    if key_format == '1':
        inputs['magnitudes'] = inputs.pop('input')
        inputs['att_mask']   = inputs.pop('mask_in')
        del inputs['mask_out']
        if num_cls is None:
            outputs['magnitudes'] = outputs.pop('target')
            outputs['probed_mask'] = outputs.pop('mask_out')
            
    if return_ids:
        return outputs, batch['lcid']
        
    return inputs, outputs