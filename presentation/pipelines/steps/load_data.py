import tensorflow as tf

from src.data.zero import pretraining_pipeline


def build_tf_data_loader(data_path, probed, random_same, num_cls=None, batch_size=5):

	train_loader = pretraining_pipeline(os.path.join(data_path, 'train'),
	                                batch_size=batch_size, 
	                                window_size=model_config['window_size'],
	                                shuffle=True,
	                                sampling=False,
	                                repeat=1,
	                                msk_frac=probed,
	                                rnd_frac=random_same,
	                                same_frac=random_same,
	                                num_cls=num_cls)
	valid_loader = pretraining_pipeline(os.path.join(data_path, 'val'),
	                                batch_size=batch_size,
	                                window_size=model_config['window_size'],
	                                shuffle=False,
	                                sampling=False,
	                                msk_frac=probed,
	                                rnd_frac=random_same,
	                                same_frac=random_same,
	                                num_cls=num_cls)
	test_loader = pretraining_pipeline(os.path.join(data_path, 'test'),
	                                batch_size=batch_size,
	                                window_size=model_config['window_size'],
	                                shuffle=False,
	                                sampling=False,
	                                msk_frac=probed,
	                                rnd_frac=random_same,
	                                same_frac=random_same,
	                                num_cls=num_cls)

	return {
		'train': train_loader,
		'validation': valid_loader,
		'test_loader': test_loader
	}