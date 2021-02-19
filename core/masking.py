import tensorflow as tf

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32, name='PaddingMask')
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = tf.math.subtract(1., 
                tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='LowerTriangular'),
                name='LookaHeadMask')
    return mask  # (seq_len, seq_len)
    
def create_masks(seq):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_look_ahead_mask(tf.shape(seq)[1])
    combined_mask = tf.maximum(enc_padding_mask, look_ahead_mask, name='CombinedMask')
    return combined_mask