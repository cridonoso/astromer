import tensorflow as tf

@tf.function
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

@tf.function
def add_segment_to_tensor(tensor_0, tensor_1, pivot, rnd_seq_size, cls_tkn=-99., sep_tkn=-98.):
    """
        Add a random segment in a tensor sequence. (user for NSP task)
        Cut a window of length <rnd_seq_size> from tensor_1, then
        paste it in tensor_0 starting from the pivot position

    Args:
        tensor_0: target sequence (or matrix)
        tensor_1: random sequence used to get the random window (or matrix)
        pivot: position to put the random window
        rnd_seq_size: size of the random window
        cls_tkn: [CLS] token to predict if the segment is random or not
        sep_tkn: [SEP] token to distinguish between segments

    Returns:
        type: a randomized tensor of size len(tensor_0) + 3, where 3
              comes from adding [CLS] and 2x[SEP]

    """

    inp_length = tf.shape(tensor_0)[0]
    inp_dim = tf.shape(tensor_0)[-1]

    rand_msk = tf.range(pivot, pivot+rnd_seq_size)
    rand_msk = tf.one_hot(rand_msk, inp_length)
    rand_msk = tf.reduce_sum(rand_msk, 0)
    rand_msk = tf.expand_dims(rand_msk, 1)
    rand_msk = tf.tile(rand_msk, [1, inp_dim])

    tensor_1 = tensor_1 * rand_msk
    tensor_0 = tensor_0 * (1.-rand_msk)
    tensor_2 = tensor_0 + tensor_1

    a = tf.range(0, pivot)
    b = tf.range(pivot+1, pivot+rnd_seq_size+1)
    c = tf.range(pivot+rnd_seq_size+2, inp_length+2)

    indices = tf.concat([a,b,c], axis=0)
    indices = tf.expand_dims(indices, 1)

    base     = tf.zeros([inp_length+2, inp_dim]) + sep_tkn
    tensor_3 = tf.tensor_scatter_nd_update(base,
                                           indices,
                                           tensor_2)

    tensor_3 = tf.concat([tf.tile([[cls_tkn]], [1,inp_dim]), tensor_3], 0)
    return tensor_3

# @tf.function
def randomize_segment(batch, random_sample, frac=.5, sep_token=-98, cls_token=-99):

    nsp_label = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    # IF nsp is 1 then don't change the sequence but add the tokens anyways
    if tf.cast(nsp_label, tf.bool):
        random_sample = batch

    inp_length = tf.shape(batch['input_modified'])[0]
    size = tf.cast(inp_length, tf.float32)*frac
    print(size)
    rnd_seq_size = tf.cast(size, tf.int32)

    # Where the random segment start
    pivot = tf.random.uniform(shape=(),
                              minval=0,
                              maxval=inp_length-rnd_seq_size,
                              dtype=tf.int32)

    # Put a random segment in the input_modified sequence (masked input)
    batch['input_modified'] = add_segment_to_tensor(batch['input_modified'],
                                                    random_sample['input_modified'],
                                                    pivot,
                                                    rnd_seq_size)

    # Add tokens to the original times
    original_times  = tf.slice(batch['input'], [0,0],[-1, 1])
    original_times  = add_segment_to_tensor(original_times,
                                            original_times,
                                            pivot,
                                            rnd_seq_size)

    # Put a random segment in the original input
    batch['input'] = add_segment_to_tensor(batch['input'],
                                           random_sample['input'],
                                           pivot,
                                           rnd_seq_size)
    # keep the original times
    batch['input'] = tf.concat([original_times,
                                tf.slice(batch['input'], [0,1], [-1,-1])],
                                1)

    # Add 0s to mask_in to be included in the self-attention op
    batch['mask']  = add_segment_to_tensor(batch['mask'],
                                           random_sample['mask'],
                                           pivot,
                                           rnd_seq_size,
                                           cls_tkn=0.,
                                           sep_tkn=0.)

    # Add 0s to mask_in to be included in the self-attention op
    batch['mask_in']  = add_segment_to_tensor(batch['mask_in'],
                                              random_sample['mask_in'],
                                              pivot,
                                              rnd_seq_size,
                                              cls_tkn=0.,
                                              sep_tkn=0.)

    # Add 0s to mask_out to exclude tokens from the loss function
    batch['mask_out'] = add_segment_to_tensor(batch['mask_out'],
                                              random_sample['mask_out'],
                                              pivot,
                                              rnd_seq_size,
                                              cls_tkn=0.,
                                              sep_tkn=0.)
    batch['nsp_label'] = nsp_label
    return batch


def nsp_dataset(dataset, prob=.5, frac=.5, buffer_shuffle=5000):
    """
    Add random next sentence with probability 1-prob

    Args:
        dataset: tf.dataset with light curves sequences (use after masking)
        prob: probability of not adding random segment
        frac: fraction of the sequence to change. The random window size depends on this.
        buffer_shuffle: buffer for shuffling the same dataset and get random segments

    Returns:
        dataset: tf.dataset that includes <nsp_label>. After use this function,
                 all sequences will be modified to incorporate [CLS] and [SEP] tokens
                 i.e., input, input_modified, mask, mask_in, mask_out....

    """

    shuffle_dataset = dataset.shuffle(buffer_shuffle)

    dataset = tf.data.Dataset.zip((dataset, shuffle_dataset))
    dataset = dataset.map(lambda x, y: randomize_segment(x,
                                                         random_sample=y,
                                                         frac=frac))
    return dataset
