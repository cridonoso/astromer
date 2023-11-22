import tensorflow as tf
import numpy as np
import toml
import os

from src.losses             import custom_rmse
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras        import Model
import tensorflow as tf
from pathlib import Path
import joblib

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
				 pe_base=1000,
				 pe_dim=128,
				 pe_c=1,
				 window_size=100,
				 batch_size=None,
				 encoder_mode='normal',
				 average_layers=False,
				 mask_format='first'):

	placeholder = build_input(window_size)

	encoder = Encoder(window_size=window_size,
					  num_layers=num_layers,
					  num_heads=num_heads,
					  head_dim=head_dim,
					  mixer_size=mixer_size,
					  dropout=dropout,
					  pe_base=pe_base,
					  pe_dim=pe_dim,
					  pe_c=pe_c,
					  average_layers=average_layers,
					  mask_format=mask_format,
					  name='encoder')

	x = encoder(placeholder)
	x = RegLayer(name='regression')(x)

	return Model(inputs=placeholder, outputs=x, name="ASTROMER-1")


@tf.function
def train_step(model, x, y, optimizer, **kwargs):
	with tf.GradientTape() as tape:
		x_pred = model(x, training=True)
		rmse = custom_rmse(y_true=y['magnitudes'],
						  y_pred=x_pred,
						  mask=y['probed_mask'])
		r2_value = custom_r2(y['magnitudes'], x_pred, y['probed_mask'])

	grads = tape.gradient(rmse, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}


@tf.function
def test_step(model, x, y, return_pred=False, **kwargs):
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

def restore_model(model_folder, mask_format=None):
    with open(os.path.join((model_folder,"config.toml")), 'r') as f:
        model_config = toml.load(f)
    
    if 'mask_format' in model_config:
        mask_format = model_config['mask_format']

    astromer = get_ASTROMER(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'],
                            encoder_mode=model_config['encoder_mode'],
                            average_layers=model_config['avg_layers'],
                            mask_format=mask_format)
    print(mask_format)
    print('[INFO] LOADING PRETRAINED WEIGHTS')
    astromer.load_weights(os.path.join(model_folder, 'weights', 'weights'))

    return astromer, model_config

def get_embeddings(astromer, dataset, model_config):
    encoder = astromer.get_layer('encoder')
    embeddings = []
    for x, y in dataset:
        Z = encoder(x)
        embeddings.append(Z.numpy())
        
    max_seq_len = model_config.window_size
    embedding_dim = model_config.mixer

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def save_embeddings(embeddings, output_path, file_name):
    extension = ".joblib"
    path = Path(f"{output_path}/{file_name}{extension}")
    with open(path, "wb") as f:
        joblib.dump(embeddings, f)
    print(f"[INFO] Successfully stored embeddings at path {path}")
