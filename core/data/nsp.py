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

def randomize_segment(batch, frac, prob=.5):
    input = batch['input_modified']
    times = tf.slice(batch['input'], [0,0,0], [-1,-1,1])
    dt    = times[:, 1:, :] - times[:, :-1, :]

    n_obs = tf.expand_dims(tf.reduce_sum(batch['mask'], 1), 1)
    mean_dt, std_dt = get_mean_and_std(dt, n_obs)


    inp_shape = tf.shape(input)
    n_random = tf.math.ceil(tf.cast(inp_shape[1], tf.float32)*frac)

    prob = tf.random.categorical(tf.math.log([[prob, 1.-prob]]), 1)
    if tf.squeeze(prob):
        first_part = tf.slice(input, [0,0,0], [-1,n_random,-1])

    else:
        print('0')

    return batch


def nsp_dataset(dataset, frac=.5):

    # dataset = dataset.map(lambda x: randomize_segment(x, frac=frac))

    for b in dataset:
        randomize_segment(b, frac=frac)
        break

    return dataset
