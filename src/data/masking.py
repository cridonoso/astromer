import tensorflow as tf
from src.data.preprocessing import calculate_iqr
from src.data.zero import set_random


def get_probed(input_dict, probed, njobs):

    input_shape = tf.shape(input_dict['input'])

    if probed == 1.:
        probed_mask = tf.ones([input_shape[0], input_shape[1]]) * input_dict['mask']
        input_dict['probed_mask']  = probed_mask
        input_dict['att_mask'] = 1. - probed_mask
        return input_dict

    nprobed = tf.multiply(tf.cast(input_shape[1], tf.float32), probed)
    nprobed = tf.cast(nprobed, tf.int32)
    random_integers = tf.range(input_shape[1], dtype=tf.int32)
    indices = tf.map_fn(lambda x: tf.random.shuffle(random_integers), 
                                      tf.range(input_shape[0]),
                                      parallel_iterations=njobs)
    indices = tf.slice(indices, [0, 0], [-1, nprobed])
    random_mask = tf.one_hot(indices, input_shape[1])
    random_mask = tf.reduce_sum(random_mask, 1)

    input_dict['probed_mask'] = random_mask * input_dict['mask']
    att_mask = (1 - input_dict['mask']) + random_mask 
    att_mask = tf.minimum(att_mask, 1)
    input_dict['att_mask'] = att_mask

    return input_dict

def get_probed_astrospec(input_dict,exclusive_frac,non_exclusive_frac,iqr_threshold,random_fraction=0.2,same_fraction=0.2,njobs=None):
    
    input_sequence=input_dict['input']
    moving_median_sequence=input_dict['moving_median_sequence']
    
    steps = tf.shape(input_sequence)[0]
    indices = tf.range(steps)
    
    input_flux=tf.slice(input_sequence,[0,1],[-1,1])
    moving_median_flux=tf.slice(moving_median_sequence,[0,1],[-1,1])
    
    squeezed_input_flux=tf.squeeze(input_flux)
    squeezed_moving_median_flux=tf.squeeze(moving_median_flux)
    
    iqr=calculate_iqr(squeezed_moving_median_flux)
    
    exclusive_emission=tf.cast(((squeezed_input_flux)>(squeezed_moving_median_flux+iqr_threshold*iqr)),tf.int32)
    exclusive_absorption=tf.cast(((squeezed_input_flux)<(squeezed_moving_median_flux-iqr_threshold*iqr)),tf.int32)
    exclusive_zone=exclusive_emission+exclusive_absorption
    
    
    emission_startidx_size=startidx_size_pair(exclusive_emission)
    absorption_startidx_size=startidx_size_pair(exclusive_absorption)
    
    
    exclusion_startidx_size=tf.concat([emission_startidx_size,absorption_startidx_size],axis=0)
    shuffled_exclusive_startidx_size=tf.random.shuffle(exclusion_startidx_size)
    
    num_rows_exclusive=tf.cast(tf.multiply(tf.cast(tf.shape(exclusion_startidx_size)[0], tf.float32), exclusive_frac),tf.int32)
    selected_rows_exclusive = tf.slice(shuffled_exclusive_startidx_size, [0, 0], [num_rows_exclusive, -1])
    
    exclusive_mask=tf.zeros_like(exclusive_zone)
    exclusive_mask=update_mask(exclusive_mask,selected_rows_exclusive)
    
    
    non_exclusive_zone=tf.math.logical_not(tf.cast(exclusive_zone,tf.bool))
    non_exclusive_zone=tf.cast(non_exclusive_zone,tf.int32)
    
    
    non_exclusive_startidx_size=startidx_size_pair(non_exclusive_zone)
    shuffled_non_exclusive_startidx_size=tf.random.shuffle(non_exclusive_startidx_size)
    
    num_rows_non_exclusive=tf.cast(tf.multiply(tf.cast(tf.shape(non_exclusive_startidx_size)[0], tf.float32), non_exclusive_frac),tf.int32)
    selected_rows_non_exclusive = tf.slice(shuffled_non_exclusive_startidx_size, [0, 0], [num_rows_non_exclusive, -1])
    
    non_exclusive_mask=tf.zeros_like(non_exclusive_zone)
    non_exclusive_mask=update_mask(non_exclusive_mask,selected_rows_non_exclusive)
    
    exclusive_mask=tf.cast(exclusive_mask,dtype=tf.float32)
    non_exclusive_mask=tf.cast(non_exclusive_mask,dtype=tf.float32)

    probed_mask=exclusive_mask+non_exclusive_mask
    
    input_flux,attention_mask=set_random(input_flux,probed_mask,input_flux,same_fraction,name="set_same")
    input_flux,attention_mask=set_random(input_flux,attention_mask,tf.random.shuffle(input_flux),random_fraction,name="set_random")
    
    input_dict['probed_mask']=probed_mask
    input_dict['att_mask']=attention_mask
    
    return input_dict
    
def create_mask(pre_mask, n_elements):
    indices = tf.where(pre_mask)
    indices = tf.random.shuffle(indices)
    indices = tf.slice(indices, [0, 0], [n_elements, -1])

    mask = tf.one_hot(indices, tf.shape(pre_mask)[0], dtype=tf.int32)
    mask = tf.reduce_sum(mask, 0)
    mask = tf.reshape(mask, [tf.shape(pre_mask)[0]])
    
    return mask

def add_random(input_dict, random_frac, njobs):
    """ Add random observations to each sequence
        
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
    
    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, ],
              'label' :(),
              'input_modified': [None, None],
              'att_mask': [None, None],
              'probed_mask': [None, None],
              'mean_values':[None, None]}

    return dataset, shapes

# ======================================
# ======================================
# ======================================
# ======================================
# ======================================

# def get_probed(input_dict, probed, njobs):

#     input_shape = tf.shape(input_dict['input']) # (batch x steps x 3)

#     if probed == 1.:
#         probed_mask = tf.ones([input_shape[0], input_shape[1]]) * input_dict['mask']
#         input_dict['probed_mask']  = probed_mask
#         input_dict['att_mask'] = 1. - probed_mask
#         return input_dict
    
    
#     nprobed = tf.multiply(tf.cast(input_shape[1], tf.float32), probed)
#     nprobed = tf.cast(nprobed, tf.int32)
#     random_integers = tf.range(input_shape[1], dtype=tf.int32)
    
#     indices = tf.map_fn(lambda x: tf.random.shuffle(random_integers), 
#                                       tf.range(input_shape[0]),
#                                       parallel_iterations=njobs)
#     indices = tf.slice(indices, [0, 0], [-1, nprobed])
#     random_mask = tf.one_hot(indices, input_shape[1])
#     random_mask = tf.reduce_sum(random_mask, 1)
    
#     input_dict['probed_mask'] = random_mask*input_dict['mask']
    
#     att_mask = (1 - input_dict['mask']) + random_mask 
#     att_mask = tf.minimum(att_mask, 1)
#     input_dict['att_mask'] = att_mask

#     return input_dict

# def create_mask(pre_mask, n_elements):
#     indices = tf.where(pre_mask)
#     indices = tf.random.shuffle(indices)
#     indices = tf.slice(indices, [0, 0], [n_elements, -1])

#     mask = tf.one_hot(indices, tf.shape(pre_mask)[0], dtype=tf.int32)
#     mask = tf.reduce_sum(mask, 0)
#     mask = tf.reshape(mask, [tf.shape(pre_mask)[0]])
    
#     return mask

# def add_random(input_dict, random_frac, njobs):
#     """ Add random observations to each sequence
        
#     Args:
#         random_frac (number): Fraction of probed (in decimal) to be replaced with random values
#     """ 
#     input_shape = tf.shape(input_dict['input'])
#     input_dict['input_pre_nsp'] = input_dict['input'] 

#     # ====== RANDOM MASK =====
#     n_probed = tf.reduce_sum(input_dict['probed_mask'], 1)
#     n_random = tf.math.ceil(n_probed * random_frac)
#     n_random = tf.cast(n_random, tf.int32)
#     random_mask = tf.map_fn(lambda x: create_mask(x[0], x[1]),
#                                 (input_dict['probed_mask'], n_random),
#                                 parallel_iterations=njobs,
#                                 fn_output_signature=tf.int32)

#     # ====== SAME MASK =====
#     rest = tf.cast(input_dict['probed_mask'], tf.int32) * (1-random_mask)
#     n_rest = tf.reduce_sum(rest, 1)
#     n_same = tf.math.ceil(tf.cast(n_rest, tf.float32) * random_frac)
#     n_same = tf.cast(n_same, tf.int32)

#     same_mask = tf.map_fn(lambda x: create_mask(x[0], x[1]), 
#                                   (rest, n_same),
#                                   parallel_iterations=njobs,
#                                   fn_output_signature=tf.int32)

#     # ===== REPLACEMENT ==== 
#     random_replacement = tf.random.shuffle(tf.transpose(input_dict['input'], [1, 0, 2]))
#     random_replacement = tf.transpose(random_replacement, [1, 0, 2])
#     random_replacement = random_replacement * tf.cast(tf.expand_dims(random_mask, -1), tf.float32) * [0., 1., 1.]

#     # Mask refering to observations that do not change
#     keep_mask = tf.expand_dims(1 - random_mask, -1)
#     keep_mask = tf.tile(keep_mask, [1, 1, input_shape[-1]-1])
#     keep_mask = tf.concat([tf.zeros([input_shape[0], input_shape[1], 1], dtype=tf.int32), keep_mask], 2)
#     keep_mask = tf.abs([1, 0, 0] - keep_mask)

#     # Part of the input we mantain
#     keep_input = input_dict['input'] * tf.cast(keep_mask, tf.float32)

#     # Replacing original input with the randomized one
#     input_dict['input']  = random_replacement + keep_input

#     # Attention mask is 1 when masked. 
#     # Random mask is 1 for masked observations selected to be randomized
#     # then,
#     att_mask    = tf.cast(input_dict['att_mask'], tf.bool)
#     random_mask = tf.cast(random_mask, tf.bool)
#     same_mask   = tf.cast(same_mask, tf.bool)

#     att_mask = tf.math.logical_xor(att_mask, random_mask)
#     att_mask = tf.math.logical_xor(att_mask, same_mask)

#     input_dict['att_mask'] = tf.cast(att_mask, tf.float32)

    return input_dict

def startidx_size_pair(mask_sequence):
	windows = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
	start = 0
	window_size = 0

	for i in tf.range(tf.shape(mask_sequence)[0]):
		if mask_sequence[i] == 1:
			if window_size == 0:
				start = i
			window_size += 1
		else:
			if window_size > 1:
				windows = windows.write(windows.size(), (start, window_size))
			window_size = 0
	return windows.stack()

def update_mask(mask,selected_rows):
    for k in tf.range(tf.shape(selected_rows)[0]):
        indices_selected=tf.range(selected_rows[k][0],selected_rows[k][0]+selected_rows[k][1])
        updates = tf.ones(selected_rows[k][1], dtype=mask.dtype)
        reshaped_indices=tf.reshape(indices_selected,[-1,1])
        mask = tf.tensor_scatter_nd_update(mask, reshaped_indices, updates)
    return mask