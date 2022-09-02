import tensorflow as tf


def get_mean_and_std(tensor, N):
    """
    Return the mean and standard deviation of a padded batch.
    Args:
        tensor : tensor of size (n_samples x length x 1)
        N: tensor of dimension (n_samples x 1) with the real lengths of light curves.
           Notice, when there is no padding N == length
    Returns:
        tuple: A tuple of tensors of dimension (n_samples x 1)
               containing the mean and standard deviation of each sample in the batch.
    """
    mean_ = tf.math.divide_no_nan(tf.reduce_sum(tensor, 1), N)
    diff = tf.pow(tf.squeeze(tensor) - mean_, 2)
    diff = tf.reduce_sum(diff, 1)
    std_ = tf.math.sqrt(
                tf.math.divide_no_nan(
                        tf.expand_dims(diff, 1), N)
                )
    return mean_, std_

def get_random(tensor_0, tensor_1, pivot, rnd_seq_size):
    inp_length = tf.shape(tensor_0)[0]

    rand_msk = tf.range(pivot, pivot+rnd_seq_size)
    rand_msk = tf.one_hot(rand_msk, inp_length)
    rand_msk = tf.reduce_sum(rand_msk, 0)
    rand_msk = tf.expand_dims(rand_msk, 1)

    tensor_1 = tensor_1 * rand_msk
    tensor_0 = tensor_0 * (1.-rand_msk)
    tensor_2 = tensor_0 + tensor_1

    a = tf.range(0, pivot)
    b = tf.range(pivot+1, pivot+rnd_seq_size+1)
    c = tf.range(pivot+rnd_seq_size+2, inp_length+2)
    indices = tf.concat([a,b,c], axis=0)
    indices = tf.expand_dims(indices, 1)
    base     = tf.zeros([inp_length+2, 1])
    tensor_3 = tf.tensor_scatter_nd_add(base,
                                        indices,
                                        tensor_2)
    return tensor_3

def add_tokens(tensor, sep_token, cls_token):
    msk     = tf.cast(tensor, tf.bool)
    msk     = tf.cast(tf.logical_not(msk), tf.float32)
    tokens  = msk*tf.cast(sep_token, tf.float32)
    scatter = tensor+tokens
    scatter = tf.concat([[[cls_token]], scatter], 0)
    return scatter

def randomize_segment(batch, random_sample, frac=.5, prob=.5, sep_token=-98, cls_token=-99):

    nsp_label = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    inp_length = tf.shape(batch['input_modified'])[0]
    rnd_seq_size = tf.cast(tf.cast(inp_length, tf.float32)*frac, tf.int32)

    pivot = tf.random.uniform(shape=(),
                              minval=0,
                              maxval=inp_length-rnd_seq_size,
                              dtype=tf.int32)

    random_sequence = tf.random.shuffle(random_sample['input_modified'])

    random_input = get_random(batch['input_modified'],
                              random_sequence,
                              pivot,
                              rnd_seq_size)
    random_input = add_tokens(random_input, sep_token, cls_token)

    original_input  = tf.slice(random_sample['input'], [0,1],[-1, 1])
    random_original = get_random(original_input,
                                 random_sequence,
                                 pivot,
                                 rnd_seq_size)
    random_original = add_tokens(random_original, sep_token, cls_token)

    print(random_original)
    # indices = tf.constant([[4], [3], [1], [7]])
    # print(indices.shape)
    # updates = tf.constant([9, 10, 11, 12])
    # shape = tf.constant([inp_length])
    # scatter = tf.scatter_nd(indices, tf.squeeze(), shape=shape)

    # print(scatter)

    return batch


def nsp_dataset(dataset, prob=.5, frac=.5):

    #dataset = dataset.map(lambda x: randomize_segment(x, frac=frac))

    random_sample = dataset.shuffle(5000)

    for b, r in zip(dataset, random_sample):
        randomize_segment(b, random_sample=r, frac=frac, prob=prob)
        break

    return dataset
