''''
ASTROMER + Skip connections + Next Segment Prediction
'''
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from src.layers  import Encoder, ConcatEncoder, TransformLayer, RegLayer
from src.losses  import custom_rmse, custom_bce
from src.metrics import custom_r2, custom_acc


def build_input(window_size):
	window_size = window_size + 1
	
	magnitudes  = Input(shape=(window_size, 1),
				  batch_size=None,
				  name='magnitudes')
	times       = Input(shape=(window_size, 1),
				  batch_size=None,
				  name='times')
	att_mask    = Input(shape=(window_size, 1),
				  batch_size=None,
				  name='att_mask') 
	seg_emb     = Input(shape=(window_size, 1),
				  batch_size=None,
				  name='seg_emb')

	pholder = {'magnitudes':magnitudes,
			   'times':times,
			   'att_mask':att_mask,
			   'seg_emb':seg_emb}

	return pholder

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

	if encoder_mode == 'normal':
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

	if encoder_mode == 'concat':
		encoder = ConcatEncoder(window_size=window_size,
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

	reg_layer = TransformLayer(name='regressor')
	
	x = encoder(placeholder)
	outputs = reg_layer(x)
	return Model(inputs=placeholder, outputs=outputs, name="ASTROMER_NSP")

@tf.function
def train_step(model, x, y, optimizer, rmse_factor=0.5):
	with tf.GradientTape() as tape:
		outputs = model(x, training=True)
		
		rmse = custom_rmse(y_true=y['magnitudes'],
						   y_pred=outputs['reconstruction'],
						   mask=y['probed_mask'])

		bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
		
		loss = rmse_factor*rmse + (1.-rmse_factor)*bce

		r2_value = custom_r2(y_true=y['magnitudes'], 
							 y_pred=outputs['reconstruction'], 
							 mask=y['probed_mask'])

		nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])
	
	grads = tape.gradient(loss, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
	
	return {'loss':loss,
			'rmse': rmse,
			'r_square':r2_value,
			'bce':bce,
			'acc':nsp_acc}

@tf.function
def test_step(model, x, y, rmse_factor=0.5):
	outputs = model(x, training=False)
	
	rmse = custom_rmse(y_true=y['magnitudes'],
					   y_pred=outputs['reconstruction'],
					   mask=y['probed_mask'])

	bce = custom_bce(y['nsp_label'], outputs['nsp_label'])
	
	loss = rmse_factor*rmse + (1.-rmse_factor)*bce

	r2_value = custom_r2(y_true=y['magnitudes'], 
						 y_pred=outputs['reconstruction'], 
						 mask=y['probed_mask'])

	nsp_acc  = custom_acc(y['nsp_label'], outputs['nsp_label'])
	
	return {'loss':loss,
			'rmse': rmse,
			'r_square':r2_value,
			'bce':bce,
			'acc':nsp_acc}




