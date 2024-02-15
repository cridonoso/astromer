import tensorflow as tf 
import toml 
import os

from src.models.astromer_0 import get_ASTROMER


def build_model(params):
	if params['arch'] == 'normal' or params['arch'] == 'base':
		model = get_ASTROMER(num_layers=params['num_layers'],
							 d_model=params['head_dim']*params['num_heads'],
							 num_heads=params['num_heads'],
							 dff=params['mixer'],
							 base=params['pe_base'],
							 rate=params['dropout'],
							 use_leak=False,
							 maxlen=params['window_size'],
							 m_alpha=params['m_alpha'])

	if params['arch'] == 'skip':
		pass

	if params['arch'] == 'nsp':
		pass

	return model

def load_pt_model(pt_path):

	config_file = os.path.join(pt_path, 'config.toml')
	with open(config_file, 'r') as file:
		pt_config = toml.load(file)

	model = build_model(pt_config)

	weights_path = os.path.join(pt_path, 'weights')
	model.load_weights(weights_path).expect_partial()

	return model, pt_config