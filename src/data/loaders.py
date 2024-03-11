import tensorflow as tf
import multiprocessing
import glob
import os

from src.data.preprocessing import to_windows, standardize, min_max_scaler, nothing
from src.data.gap import set_gap, invert_mask
from src.data.masking import mask_dataset
from src.data.record import deserialize
from src.data.nsp import apply_nsp

def load_records(records_dir):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """
    record_files = glob.glob(os.path.join(records_dir, '*.record'))
    if len(record_files) == 0:
        record_files = glob.glob(os.path.join(records_dir, '*', '*.record'))
        
    raw_dataset = tf.data.TFRecordDataset(record_files)
    raw_dataset = raw_dataset.map(lambda x: deserialize(x, records_dir))
    return raw_dataset

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
                        aversion='base'):
    """
    Buildng ASTROMER input

    Args:
        batch (type): a batch of windows and their properties

    Returns:
        type: A tuple (x, y) tuple where x are the inputs and y the labels
    """

    inputs, outputs = {}, {}
    if aversion=='redux':
        inputs['input']    =  batch['input_modified']
        t_mean = tf.slice(batch['mean_values'], [0, 0], [-1, 1])
        t_mean = tf.expand_dims(t_mean, axis=1)
        times = tf.slice(batch['input'], [0,0,0], [-1,-1,1]) + t_mean
        t_min = tf.reduce_min(times, axis=1)
        t_min = tf.expand_dims(t_min, axis=1)
        t_max = tf.reduce_max(times, axis=1)
        t_max = tf.expand_dims(t_max, axis=1)
        inputs['times']    =  tf.math.divide_no_nan(times - t_min, t_max-t_min) + 0.1
        
        inputs['mask_in']  =  batch['mask_in']
        inputs['mask_out'] = tf.expand_dims(batch['mask_out'], -1)
        
        outputs['target']   =  tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['error']    =  tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['mask_out'] =  batch['mask_out']
        outputs['lcid']     = batch['lcid']


    if aversion == 'base' or aversion == 'normal':
        inputs['input']    =  batch['input_modified']
        inputs['times']    =  tf.slice(batch['input'], [0,0,0], [-1,-1,1])

        inputs['mask_in']  =  batch['mask_in']
        inputs['mask_out'] = tf.expand_dims(batch['mask_out'], -1)
        
        outputs['target']   =  tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['error']    =  tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['mask_out'] =  batch['mask_out']
        outputs['lcid'] = batch['lcid']
            
    if aversion == 'skip':
        inputs['input'] = batch['input_modified']
        times = tf.slice(batch['input'], [0,0,0], [-1,-1,1])
        inp_size = tf.shape(batch['input'])
        t0 = tf.slice(batch['input'], [0,0,0], [-1,inp_size[1]-1,1])
        t1 = tf.slice(batch['input'], [0,1,0], [-1,-1,1])
        dt = t1 - t0
        inputs['times']     = tf.concat([tf.zeros([inp_size[0], 1, 1]), dt], axis=1)
        inputs['mask_in']   = batch['mask_in']

        outputs['target']  = tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['error']       = tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['mask_out'] = batch['mask_out']
    
    if aversion == 'nsp':
        inputs['input']   = batch['nsp_magnitudes']
        inputs['times']   = batch['nsp_times']
        inputs['mask_in'] = batch['mask_in']
        inputs['seg_emb'] = tf.expand_dims(batch['seg_emb'], axis=-1)
        
        outputs['target'] = batch['target_magnitudes']
        outputs['error']  = tf.slice(batch['input'], [0,0,2], [-1,-1,1])
        outputs['original']  = tf.slice(batch['input'], [0,0,1], [-1,-1,1])
        outputs['mask_out']  = batch['mask_out']
        outputs['seg_emb']   = tf.where(inputs['seg_emb'] == -1., 1., 0.)
        outputs['nsp_label'] = batch['nsp_label']
    

    if num_cls is not None:
        outputs = tf.one_hot(batch['label'], num_cls)
    
    if return_ids:     
        outputs = tf.one_hot(batch['label'], num_cls), batch['lcid']
        
    if return_lengths:
        outputs = tf.one_hot(batch['label'], num_cls), batch['lenght']

    return inputs, outputs

def filter_fn(input_dict):
    if tf.less(tf.shape(input_dict['input'])[0], 5):
        return False
    else:
        return True
    
def get_loader(dataset,
               batch_size=None,
               window_size=200,
               probed_frac=.2,
               random_frac=.1,
               same_frac=None,
               nsp_prob=0.5,
               sampling=True,
               shuffle=False,
               repeat=1,
               num_cls=None,
               max_gap=0.2,
               normalize='zero-mean', # 'minmax'
               cache=False,
               return_ids=False,
               return_lengths=False,
               aversion='skip'):


    assert isinstance(dataset, (list, str)), '[ERROR] Invalid format'
    assert batch_size is not None, '[ERROR] Undefined batch size'

    if isinstance(dataset, list):
        dataset = load_numpy(dataset)

    if isinstance(dataset, str):
        dataset = load_records(records_dir=dataset)
    
    if shuffle:
        SHUFFLE_BUFFER = 10000
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # REPEAT LIGHT CURVES
    if repeat is not None:
        print('[INFO] Repeating dataset x{} times'.format(repeat))
        dataset = dataset.repeat(repeat)
    
    dataset = dataset.filter(filter_fn)

    # CREATE WINDOWS
    dataset = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)
    
    if normalize is None:
        dataset = dataset.map(nothing)
    if normalize == 'zero-mean':
        dataset = dataset.map(standardize)

    if normalize == 'minmax':
        dataset = dataset.map(min_max_scaler)
        
    
    print('[INFO] Loading PT task: Masking')
    dataset, shapes = mask_dataset(dataset,
                           msk_frac=probed_frac,
                           rnd_frac=random_frac,
                           same_frac=random_frac if same_frac is None else same_frac,
                           window_size=window_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    if aversion == 'nsp':
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

    # #PREFETCH BATCHES
    dataset = dataset.prefetch(2)

    return dataset
