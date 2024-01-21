import tensorflow as tf

from src.data.record import deserialize
import numpy as np
import tensorflow_probability as tfp

def standardize(batch, on='input', axis=0, return_mean=False):
    """
    Standardize input tensor given a dataset batch
    Args:
        dataset: batched dataset

    Returns:
        type: tf.Dataset
    """

    mean_value = tf.reduce_mean(tensor['input'], axis, name='mean_value')
    tensor['input'] = tensor['input'] - tf.expand_dims(mean_value, axis)

    mean_value = tf.reduce_mean(tensor['original'], axis, name='mean_value')
    tensor['original'] = tensor['original'] - tf.expand_dims(mean_value, axis)
    return tensor
    if return_mean:
        return z, mean_value
    else:
        return z

def standardize_batch(input_tensor, axis=1):
	N = tf.where(tf.abs(input_tensor) > 0, 1., 0.)
	N = tf.slice(N, [0, 0, 0], [-1, -1, 1])
	N = tf.reduce_sum(N, 1)

	input_mean = tf.math.divide_no_nan(tf.reduce_sum(input_tensor, axis), N)
	input_mean = tf.expand_dims(input_mean, axis)
	input_tensor = input_tensor - input_mean
	return input_tensor

def standardize_dataset(input_tensor, on='input'):
	input_tensor[on] = standardize_batch(input_tensor[on])
	return input_tensor

def standardize_object(input_tensor, axis=0):
	N = tf.where(tf.abs(input_tensor) > 0, 1., 0.)
	N = tf.slice(N, [0, 0], [-1,1])
	N = tf.reduce_sum(N, 0)
	
	input_mean = tf.math.divide_no_nan(tf.reduce_sum(input_tensor, axis), N)
	input_mean = tf.expand_dims(input_mean, axis)
	input_tensor = input_tensor - input_mean
	return input_tensor

def standardize_dataset_astrospec(input_tensor, on='input'):
	input_tensor[on] = standardize_object(input_tensor[on])
	return input_tensor

def min_max_scaler(tensor, axis=1, offset=1.):
	offset = tf.cast(offset, tf.float32)
	min_val = tf.reduce_min(tensor, axis=1)
	min_val = tf.expand_dims(min_val, axis=1)
	max_val = tf.reduce_max(tensor, axis=1)
	max_val = tf.expand_dims(max_val, axis=1)
	return tf.math.divide_no_nan(tensor - min_val, max_val - min_val) + offset

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

def sample_spectra(sample, max_obs, binary=True):
	'''
	Sample a random window of "max_obs" observations from the input sequence and the moving median sequence
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
		input_dict['moving_median_sequence']=tf.slice(input_dict['moving_median_sequence'], [pivot,0], [max_obs, -1])
	else:
		input_dict['input'] = tf.slice(input_dict['input'], [0,0], [serie_len, -1])
		input_dict['moving_median_sequence']=tf.slice(input_dict['moving_median_sequence'], [0,0], [serie_len, -1])

	return input_dict

def sample_spectra(sample, max_obs, binary=True):
	'''
	Sample a random window of "max_obs" observations from the input sequence and the moving median sequence
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
		input_dict['moving_median_sequence']=tf.slice(input_dict['moving_median_sequence'], [pivot,0], [max_obs, -1])
	else:
		input_dict['input'] = tf.slice(input_dict['input'], [0,0], [serie_len, -1])
		input_dict['moving_median_sequence']=tf.slice(input_dict['moving_median_sequence'], [0,0], [serie_len, -1])

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
    rest = input_dict['length']%(max_obs)

    pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs),  pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))

    input_dict['label']  = tf.tile([input_dict['label']], [len(splits)])
    input_dict['lcid']   = tf.tile([input_dict['lcid']], [len(splits)])
    input_dict['length'] = tf.tile([input_dict['length']], [len(splits)])
    input_dict['input']  = splits

    return input_dict

def get_windows_spectra(sample, max_obs, binary=True):
	if binary:
		input_dict = deserialize(sample)
	else:
		input_dict = sample

	sequence = input_dict['input']
	sequence_moving_median = input_dict['moving_median_sequence']
 
	rest = input_dict['length']%(max_obs)

	pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
	pivots = tf.concat([[0], pivots], 0)
	pivots = tf.math.cumsum(pivots)

	splits = tf.map_fn(lambda x: get_window(sequence,
											input_dict['length'],
											x,
											max_obs),  pivots,
					   infer_shape=False,
					   fn_output_signature=(tf.float32))
 
	splits_moving_median = tf.map_fn(lambda x: get_window(sequence_moving_median,
											input_dict['length'],
											x,
											max_obs),  pivots,
					   infer_shape=False,
					   fn_output_signature=(tf.float32))

	input_dict['label']  = tf.tile([input_dict['label']], [len(splits)])
	input_dict['lcid']   = tf.tile([input_dict['lcid']], [len(splits)])
	input_dict['length'] = tf.tile([input_dict['length']], [len(splits)])
	input_dict['input']  = splits
	input_dict['moving_median_sequence'] = splits_moving_median

	return input_dict

def get_windows_spectra(sample, max_obs, binary=True):
	if binary:
		input_dict = deserialize(sample)
	else:
		input_dict = sample

	sequence = input_dict['input']
	sequence_moving_median = input_dict['moving_median_sequence']
 
	rest = input_dict['length']%(max_obs)

	pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
	pivots = tf.concat([[0], pivots], 0)
	pivots = tf.math.cumsum(pivots)

	splits = tf.map_fn(lambda x: get_window(sequence,
											input_dict['length'],
											x,
											max_obs),  pivots,
					   infer_shape=False,
					   fn_output_signature=(tf.float32))
 
	splits_moving_median = tf.map_fn(lambda x: get_window(sequence_moving_median,
											input_dict['length'],
											x,
											max_obs),  pivots,
					   infer_shape=False,
					   fn_output_signature=(tf.float32))

	input_dict['label']  = tf.tile([input_dict['label']], [len(splits)])
	input_dict['lcid']   = tf.tile([input_dict['lcid']], [len(splits)])
	input_dict['length'] = tf.tile([input_dict['length']], [len(splits)])
	input_dict['input']  = splits
	input_dict['moving_median_sequence'] = splits_moving_median

	return input_dict

def to_windows(dataset,
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

        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))


    dataset = dataset.map(lambda x: {'input' :x['input'],
                                     'lcid'  :x['lcid'],
                                     'length':x['length'],
                                     'mask'  :tf.ones(tf.shape(x['input'])[0]),
                                     'label' : x['label']},
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

def to_windows_astrospec(dataset,
			   window_size=200,
			   sampling=True):
	"""
	Transform a spectra-based tf.Dataset to a windows-based one.
	Args:
		dataset: tf.Dataset (use load_records or load_numpy first)
		batch_size (integer): Number of windows per batch
		window_size: Maximum window size. window_size<=max.length from spectra
		sampling: Windows extraction strategy.
				  If True, windows are randomnly sampled from the spectra
				  If False, spectra are divided in sequential windows
				  without overlaping.
	Returns:
		type: tf.Dataset
	"""

	if sampling:
		print('[INFO] Sampling random windows')
		dataset = dataset.map(lambda x: sample_spectra(x,
												  max_obs=window_size,
												  binary=False),
							  num_parallel_calls=tf.data.experimental.AUTOTUNE)
	else:
		dataset = dataset.map(lambda x: get_windows_spectra(x,
													max_obs=window_size,
													binary=False),
							  num_parallel_calls=tf.data.experimental.AUTOTUNE)

		dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))


	dataset = dataset.map(lambda x: {'input' :x['input'],
									 'moving_median_sequence' :x['moving_median_sequence'],
									 'original' :x['input'],
									 'lcid'  :x['lcid'],
									 'length':x['length'],
									 'mask'  :tf.ones(tf.shape(x['input'])[0]),
									 'label' : x['label']},
						  num_parallel_calls=tf.data.experimental.AUTOTUNE)

	sizes = {
		'input': (window_size, None),
		'original' :(window_size, None),
		'moving_median_sequence': (window_size, None),
		'lcid': (),
		'length': (),
		'mask': (window_size),
		'label': ()
	}
	
	return dataset, sizes

def get_moving_median(input_dict,moving_window_size):
	'''
	Calculates the moving median of padded input flux.
	Args:
		input_dict: It is a dictionary containing keys: lcid, length, label, input.
		moving_window_size: The size of window to calculate moving median.
	Returns:
        A dictionary containing keys: lcid, length, label, input, moving_median_sequence.
	'''
	input_dict['moving_median_sequence']=input_dict['input']
	input_dict['input']=input_dict['input'][:,:2]
	
	spectra_wavelength=tf.reshape(tf.slice(input_dict['moving_median_sequence'],[0,0],[-1,1]),[-1])
	spectra_flux=tf.reshape(tf.slice(input_dict['moving_median_sequence'],[0,1],[-1,1]),[-1])

	left_padded_tensor=tf.fill(tf.constant([moving_window_size//2]), spectra_flux[0])
	right_padded_tensor=tf.fill(tf.constant([moving_window_size//2]), spectra_flux[-1])
	spectra_flux_padded = tf.concat([left_padded_tensor, spectra_flux, right_padded_tensor], axis=0)

	moving_medians = tf.TensorArray(tf.float32, size=tf.shape(spectra_flux_padded)[0] - moving_window_size + 1)

	for i in range(moving_window_size // 2, tf.shape(spectra_flux_padded)[0] - moving_window_size // 2):
		window_tensor=spectra_flux_padded[i - moving_window_size // 2:i + moving_window_size // 2 + 1]
		mid=moving_window_size//2 + 1
		moving_medians = moving_medians.write(i - moving_window_size // 2,tf.nn.top_k(window_tensor, mid).values[-1])

	moving_medians_flux = moving_medians.stack()
	sequence=tf.transpose(tf.stack([spectra_wavelength,moving_medians_flux]))
	input_dict['moving_median_sequence']=sequence
	
	return input_dict

def calculate_iqr(sequence):
	q1=tfp.stats.percentile(sequence,25.0)
	q3=tfp.stats.percentile(sequence,75.0)
	return q3-q1    