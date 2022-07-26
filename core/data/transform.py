import tensorflow as tf 

def standardize(tensor, axis=0, return_mean=False):
    """
    Standardize a tensor subtracting the mean

    Args:
        tensor (1-dim tensorflow tensor): values
        axis (int): axis on which we calculate the mean
        return_mean (bool): output the mean of the tensor
                            turning on the original scale
    Returns:
        tensor (1-dim tensorflow tensor): standardize tensor
    """
    mean_value = tf.reduce_mean(tensor, axis, name='mean_value')
    z = tensor - tf.expand_dims(mean_value, axis)

    if return_mean:
        return z, mean_value
    else:
        return z
