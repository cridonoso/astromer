import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
import numpy as np
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


# Downloading BertTokenizer 
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
    )
tokenizers = tf.saved_model.load(model_name)

def tokenize_pairs(pt, en):

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return en[:, :-1], en[:, 1:]

def make_batches(ds, batchsize):
    BUFFER_SIZE = 20000
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(batchsize)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
 

