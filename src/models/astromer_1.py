import tensorflow as tf

from src.losses             import custom_rmse
from src.metrics            import custom_r2
from tensorflow.keras.layers import Input, Layer, Dense
from tensorflow.keras        import Model
import tensorflow as tf

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
				 average_layers=False):

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
	return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}


@tf.function
def test_step(model, x, y):
	x_pred = model(x, training=True)
	rmse = custom_rmse(y_true=y['magnitudes'],
					  y_pred=x_pred,
					  mask=y['probed_mask'])
	r2_value = custom_r2(y['magnitudes'], x_pred, y['probed_mask'])
	return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}
