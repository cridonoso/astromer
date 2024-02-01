import tensorflow as tf 
import toml 
import os

from src.models.astromer_0 import get_ASTROMER


def build_model(arch='base', **params):

	if arch == 'base':
		model = get_ASTROMER(**params)

	if arch == 'skip':
		pass

	if arch == 'nsp':
		pass

	return model

def load_pt_model(pt_path):

	config_file = os.path.join(pt_path, 'config.toml')
	with open(config_file, 'r') as file:
		pt_config = toml.load(file)

	print(pt_config)