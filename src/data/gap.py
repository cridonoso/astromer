import tensorflow as tf


def set_gap_prediction(dataset):

	times = tf.slice(dataset['input'], [0, 0, 0], [-1, -1, 1])
	print(times)
	
	return dataset