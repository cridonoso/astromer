import tensorflow as tf
import toml
import os

from src.losses             import custom_rmse
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras        import Model

from src.layers import Encoder, RegLayer

def build_input(length):
	serie  = Input(shape=(length, 1),
				  batch_size=None,
				  name='input')
	times  = Input(shape=(length, 1),
				  batch_size=None,
				  name='times')
	mask   = Input(shape=(length, 1),
				  batch_size=None,
				  name='mask')

	return {'magnitudes':serie,
			'att_mask':mask,
			'times':times}

def get_ASTROMER(num_layers=2,
				 num_heads=2,
				 head_dim=64,
				 mixer_size=256,
				 dropout=0.1,
				 pe_type='APE',
				 pe_config=None,
				 pe_func_name='pe',
				 residual_type=None,
				 window_size=100,
				 batch_size=None,
				 encoder_mode='normal',
				 average_layers=False,
				 data_name=None):

	placeholder = build_input(window_size)

	encoder = Encoder(window_size=window_size,
					  num_layers=num_layers,
					  num_heads=num_heads,
					  head_dim=head_dim,
					  mixer_size=mixer_size,
					  dropout=dropout,
					  pe_type=pe_type,
					  pe_config=pe_config,
					  pe_func_name=pe_func_name,
					  residual_type=residual_type,
					  average_layers=average_layers,
					  data_name=data_name,
					  name='encoder')

	x = encoder(placeholder)
	x = RegLayer(name='regression')(x)

	return Model(inputs=placeholder, outputs=x, name="ASTROMER-1")


@tf.function
def train_step(model, x, y, optimizer):
	with tf.GradientTape() as tape:
		x_pred = model(x, training=True)
		rmse = custom_rmse(y_true=y['magnitudes'],
						   y_pred=x_pred,
						   mask=y['probed_mask'])
		r2_value = custom_r2(y['magnitudes'], x_pred, y['probed_mask'])

	grads = tape.gradient(rmse, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	#tf.print(model.trainable_weights)
	return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}


@tf.function
def test_step(model, x, y, return_pred=False):
	x_pred = model(x, training=False)
	rmse = custom_rmse(y_true=y['magnitudes'],
					   y_pred=x_pred,
					   mask=y['probed_mask'])
	r2_value = custom_r2(y['magnitudes'], x_pred, y['probed_mask'])
	if return_pred:
		return x_pred
	return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}


def predict(model, test_loader):
	n_batches = sum([1 for _, _ in test_loader])
	print('[INFO] Processing {} batches'.format(n_batches))
	y_pred = tf.TensorArray(dtype=tf.float32, size=n_batches)
	y_true = tf.TensorArray(dtype=tf.float32, size=n_batches)
	masks  = tf.TensorArray(dtype=tf.float32, size=n_batches)
	times  = tf.TensorArray(dtype=tf.float32, size=n_batches)
	for index, (x, y) in enumerate(test_loader):
		outputs = test_step(model, x, y, return_pred=True)
		y_pred = y_pred.write(index, outputs)
		y_true = y_true.write(index, y['magnitudes'])
		masks  = masks.write(index, y['probed_mask'])
		times = times.write(index, x['times'])

	y_pred = tf.concat([times.concat(), y_pred.concat()], axis=2)
	y_true = tf.concat([times.concat(), y_true.concat()], axis=2)
	return y_pred, y_true, masks.concat()

def restore_model(model_folder, data_name):
	with open(os.path.join(model_folder, 'config.toml'), 'r') as f:
		model_config = toml.load(f)

	if 'residual_type' not in model_config:
		model_config['residual_type'] = None

	with open(os.path.join(model_folder, 'pe_config.toml'), 'r') as f:
		pe_config = toml.load(f)

	astromer = get_ASTROMER(num_layers=model_config['num_layers'],
							num_heads=model_config['num_heads'],
							head_dim=model_config['head_dim'],
							mixer_size=model_config['mixer'],
							dropout=model_config['dropout'],
							pe_type=model_config['pe_type'],
							pe_config=pe_config,
							pe_func_name=model_config['pe_func_name'],
							residual_type=model_config['residual_type'],
							window_size=model_config['window_size'],
							encoder_mode=model_config['encoder_mode'],
							average_layers=model_config['avg_layers'],
							data_name=data_name)

	print('[INFO] LOADING PRETRAINED WEIGHTS')
	astromer.load_weights(os.path.join(model_folder, 'weights', 'weights'))

	return astromer, model_config

def restore_ft_model(model_folder, data_name):
	with open(os.path.join(model_folder, 'config.toml'), 'r') as f:
		model_config = toml.load(f)

	#if 'residual_type' not in model_config:
	#	model_config['Pretraining']['residual_type'] = None

	with open(os.path.join(model_folder, 'pe_config.toml'), 'r') as f:
		pe_config = toml.load(f)

	astromer = get_ASTROMER(num_layers=model_config['Pretraining']['num_layers'],
							num_heads=model_config['Pretraining']['num_heads'],
							head_dim=model_config['Pretraining']['head_dim'],
							mixer_size=model_config['Pretraining']['mixer'],
							dropout=model_config['Pretraining']['dropout'],
							pe_type=model_config['Pretraining']['pe_type'],
							pe_config=pe_config,
							pe_func_name=model_config['Pretraining']['pe_func_name'],
							window_size=model_config['Pretraining']['window_size'],
							encoder_mode=model_config['Pretraining']['encoder_mode'],
							average_layers=model_config['Pretraining']['avg_layers'],
							data_name=data_name)

	print('[INFO] LOADING PRETRAINED WEIGHTS')
	astromer.load_weights(os.path.join(model_folder, 'weights', 'weights'))

	return astromer, model_config