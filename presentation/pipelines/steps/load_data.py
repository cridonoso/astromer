import tensorflow as tf
import pandas as pd
import os

from src.data.zero import pretraining_pipeline


def build_tf_data_loader(data_path, params, batch_size=5, clf_mode=False):

	if clf_mode:
		num_cls = pd.read_csv(os.path.join(data_path, 'objects.csv')).shape[0]
	else:
		num_cls = None

	train_loader = pretraining_pipeline(os.path.join(data_path, 'train'),
	                                batch_size=batch_size, 
	                                window_size=params['window_size'],
	                                shuffle=True,
	                                sampling=False,
	                                repeat=1,
	                                msk_frac=params['probed'],
	                                rnd_frac=params['rs'],
	                                same_frac=params['rs'],
	                                num_cls=num_cls)
	valid_loader = pretraining_pipeline(os.path.join(data_path, 'val'),
	                                batch_size=batch_size,
	                                window_size=params['window_size'],
	                                shuffle=False,
	                                sampling=False,
	                                msk_frac=params['probed'],
	                                rnd_frac=params['rs'],
	                                same_frac=params['rs'],
	                                num_cls=num_cls)
	test_loader = pretraining_pipeline(os.path.join(data_path, 'test'),
	                                batch_size=batch_size,
	                                window_size=params['window_size'],
	                                shuffle=False,
	                                sampling=False,
	                                msk_frac=params['probed'],
	                                rnd_frac=params['rs'],
	                                same_frac=params['rs'],
	                                num_cls=num_cls)

	return {
		'train': train_loader,
		'validation': valid_loader,
		'test_loader': test_loader
	}