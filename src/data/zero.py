import tensorflow as tf
import glob
import os

@tf.function
def reshape_mask(mask):
    ''' Reshape Mask to match attention dimensionality '''
    with tf.name_scope("reshape_mask") as scope:
        return mask[:, tf.newaxis, tf.newaxis, :, 0]

@tf.function
def get_masked(tensor, frac=0.15):
    """ Add [MASK] values to be predicted
    Args:
        tensor : tensor values
        frac (float, optional): percentage for masking [MASK]
    Returns:
        binary tensor: a time-distributed mask
    """
    with tf.name_scope("get_masked") as scope:
        steps = tf.shape(tensor)[0] # time steps
        nmask = tf.multiply(tf.cast(steps, tf.float32), frac)
        nmask = tf.cast(nmask, tf.int32, name='nmask')

        indices = tf.range(steps)
        indices = tf.random.shuffle(indices)
        indices = tf.slice(indices, [0], [nmask])

        mask = tf.reduce_sum(tf.one_hot(indices, steps), 0)
        mask = tf.minimum(mask, tf.ones_like(mask))
        return mask

@tf.function
def set_random(serie_1, mask_1, serie_2, rnd_frac, name='set_random'):
    """ Add Random values in serie_1
    Note that if serie_2 == serie_1 then it replaces the true value
    Args:
        serie_1: current serie
        mask_1 : mask containing the [MASKED]-indices from serie_1
        serie_2: random values to be placed within serie_1
        rnd_frac (float): fraction of [MASKED] to be replaced by random
                          elements from serie_2
    Returns:
        serie_1: serie_1 with random values
    """
    with tf.name_scope(name) as scope:
        nmasked = tf.reduce_sum(mask_1)
        nrandom = tf.multiply(nmasked, rnd_frac, name='mulscalar')
        nrandom = tf.cast(tf.math.ceil(nrandom), tf.int32)

        mask_indices = tf.where(mask_1)
        mask_indices = tf.random.shuffle(mask_indices)
        mask_indices = tf.reshape(mask_indices, [-1])
        mask_indices = tf.slice(mask_indices, [0], [nrandom])

        rand_mask = tf.one_hot(mask_indices, tf.shape(mask_1)[0])
        rand_mask = tf.reduce_sum(rand_mask, 0)
        rand_mask = tf.minimum(rand_mask, tf.ones_like(rand_mask))
        rand_mask = tf.expand_dims(rand_mask, 1)
        rand_mask = tf.tile(rand_mask, [1, tf.shape(serie_2)[-1]])
        
        len_s1 = tf.minimum(tf.shape(serie_2)[0],
                            tf.shape(rand_mask)[0])

        serie_2 = tf.slice(serie_2, [0,0], [len_s1, -1])

        rand_vals = tf.multiply(serie_2, rand_mask, name='randvalsmul')

        keep_mask = tf.math.floor(tf.math.cos(rand_mask))

        serie_1 = tf.multiply(serie_1, keep_mask, name='seriemul')

        keep_mask = tf.slice(keep_mask, [0,0], [-1,1])
        mask_1  = tf.multiply(mask_1, tf.squeeze(keep_mask), name='maskmul2')
        serie_1 = tf.add(serie_1, rand_vals)

        return serie_1, mask_1

def mask_sample(input_dict, msk_frac, rnd_frac, same_frac, max_obs):
    '''
    OLD VERSION
    '''
    x = input_dict['input']

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    time_steps = tf.shape(seq_magn)[0]
    orig_magn = seq_magn

    # [MASK] values
    if msk_frac == 0.:
        mask_out = tf.ones(time_steps)
    else:
        mask_out = get_masked(seq_magn, msk_frac)

    # [MASK] -> Identity
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_frac,
                                   name='set_same')

    # [MASK] -> Random
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_frac,
                                   name='set_random')
    if msk_frac == 0.:
        mask_in  =  1.- mask_out

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out,  1-mask_fill], 0)
        mask_in   = tf.concat([mask_in,     mask_fill], 0)
        seq_magn  = tf.concat([seq_magn,  1-mask_fill], 0)
        seq_time  = tf.concat([seq_time,  1-mask_fill], 0)
        orig_magn = tf.concat([orig_magn, 1-mask_fill], 0)
        input_dict['mask'] =  tf.concat([input_dict['mask'],
                                        1-tf.reshape(mask_fill, [tf.shape(mask_fill)[0]])], 0)

        reshaped_mask = tf.zeros([max_obs - time_steps,
                                  tf.shape(input_dict['input'])[-1]],
                                  dtype=tf.float32)
        input_dict['input'] = tf.concat([input_dict['input'], reshaped_mask], 0)

    input_dict['input_modified'] = seq_magn
    input_dict['mask_in']  = mask_in
    input_dict['mask_out'] = mask_out

    return input_dict

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

def sample_lc(sample, max_obs, binary=True):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    if binary:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    serie_len = tf.shape(input_dict['input'])[0]

    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-max_obs+1,
                                  dtype=tf.int32)

        input_dict['input'] = tf.slice(input_dict['input'], [pivot,0], [max_obs, -1])
    else:
        input_dict['input'] = tf.slice(input_dict['input'], [0,0], [serie_len, -1])

    return input_dict

def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length-max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced

def get_windows(sample, max_obs, binary=True):
    if binary:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    sequence = input_dict['input']

    #rest = input_dict['length']%max_obs
    #pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
    
    num_full_obs = input_dict['length'] // max_obs
    if input_dict['length'] % max_obs == 0:
        pivots = tf.tile([max_obs], [num_full_obs - 1])  # Ajuste aquí para evitar el último split redundante
    else:
        pivots = tf.tile([max_obs], [num_full_obs])

    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs),  pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))
    
    y        = tf.tile([input_dict['label']], [len(splits)])
    ids      = tf.tile([input_dict['lcid']], [len(splits)])
    orig_len = tf.tile([input_dict['length']], [len(splits)])

    return splits, y, ids, orig_len

def to_windows(dataset,
               batch_size=None,
               window_size=200,
               sampling=True):
    """
    Transform a lightcurves-based tf.Dataset to a windows-based one.
    Args:
        dataset: tf.Dataset (use load_records or load_numpy first)
        batch_size (integer): Number of windows per batch
        window_size: Maximum window size. window_size<=max.length from lightcurves
        sampling: Windows extraction strategy.
                  If True, windows are randomnly sampled from the light curves
                  If False, lightcurves are divided in sequential windows
                  without overlaping.
    Returns:
        type: tf.Dataset
    """

    if sampling:
        print('[INFO] Sampling random windows')
        dataset = dataset.map(lambda x: sample_lc(x,
                                                  max_obs=window_size,
                                                  binary=False),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: get_windows(x,
                                                    max_obs=window_size,
                                                    binary=False),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.flat_map(lambda w,x,y,z: tf.data.Dataset.from_tensor_slices((w,x,y,z)))

        dataset = dataset.map(lambda w,x,y,z: {'input':w,
                                               'label':x,
                                               'lcid':y,
                                               'length':z},
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda x: {'input' :x['input'],
                                     'lcid'  :x['lcid'],
                                     'length':x['length'],
                                     'mask'  :tf.ones(tf.shape(x['input'])[0]),
                                     'label' : x['label']},
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch_size is not None:
        dataset = dataset.padded_batch(batch_size)

    return dataset

def min_max_scaler(batch, on='input', axis=0):
    """
    Normalize input tensor given a dataset batch
    Args:
        dataset: batched dataset

    Returns:
        type: tf.Dataset
    """
    min_value = tf.reduce_min(batch['input'], axis, name='min_value')
    max_value = tf.reduce_max(batch['input'], axis, name='max_value')
    min_value = tf.expand_dims(min_value, axis)
    max_value = tf.expand_dims(max_value, axis)
    batch['input'] = tf.math.divide_no_nan(batch['input'] - min_value,
                                           max_value-min_value)
    return batch

def standardize(batch, on='input', axis=0):
    """
    Standardize input tensor given a dataset batch
    Args:
        dataset: batched dataset

    Returns:
        type: tf.Dataset
    """
    mean_value = tf.reduce_mean(batch['input'], axis, name='mean_value')
    batch['input'] = batch['input'] - tf.expand_dims(mean_value, axis)
    return batch

def deserialize(sample):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
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
    input_dict['input'] = sequence
    return input_dict

def load_records(records_dir):
    """
    Load records files containing serialized light curves.

    Args:
        records_dir (str): records folder
    Returns:
        type: tf.Dataset instance
    """
    name = '*.record'
    middle = []
    rec_paths= []
    while rec_paths == []:
        init = os.path.join(records_dir, *middle,  name)
        rec_paths = glob.glob(init)
        middle.append('*')
    
    dataset = tf.data.TFRecordDataset(rec_paths)    

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
        #print('[INFO] Repeating dataset x{} times'.format(repeat))
        dataset = dataset.repeat(repeat)
    
    # CREATE WINDOWS
    dataset = to_windows(dataset,
                         window_size=window_size,
                         sampling=sampling)
    

    if normalize == 'zero-mean':
        dataset = dataset.map(standardize)
    if normalize == 'minmax':
        dataset = dataset.map(min_max_scaler)
    
    #print('[INFO] Loading PT task: Masking')
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
    
    #if batch_size is not None:
    dataset = dataset.padded_batch(batch_size, padded_shapes=shapes)

    # FORMAT INPUT DICTONARY
    dataset = dataset.map(lambda x: format_inp_astromer(
        x,
        return_ids=return_ids,
        return_lengths=return_lengths,
        num_cls=num_cls,
        key_format=key_format), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

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
        'mask_in': batch['mask_in'],
        'mask_out':batch['mask_out']
        
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
    if return_ids:
        inputs['ids'] = batch['lcid']
    if return_lengths:
        inputs['length'] = batch['length']

    if key_format == '1':
        inputs['magnitudes'] = inputs.pop('input')
        inputs['att_mask']   = inputs.pop('mask_in')
        del inputs['mask_out']
        if num_cls is None:
            outputs['magnitudes'] = outputs.pop('target')
            outputs['probed_mask'] = outputs.pop('mask_out')

    return inputs, outputs