import tensorflow as tf

from src.data.loaders import load_data
from src.models.second import get_ASTROMER

from src.losses import custom_rmse, custom_bce
from src.metrics import custom_acc, custom_r2

tf.debugging.experimental.disable_dump_debug_info()

dataset = load_data('./data/records/macho_clean/train', nsp_prob=.4, batch_size=5000, window_size=200, repeat=1)

model = get_ASTROMER(num_layers=2,
		             num_heads=2,
		             head_dim=64,
		             mixer_size=256,
		             dropout=0.1,
		             pe_base=1000,
		             pe_dim=128,
		             pe_c=1,
		             window_size=10)

def check_tensor(tensor):
	a = tf.cast(tf.math.is_nan(tensor), tf.float32)
	return tf.reduce_sum(a)

for x in dataset:
	
	print(check_tensor(x['input']))
	# x_cls, x_rec = model(x)
	# loss = custom_rmse(y['magnitudes'], x_rec, y['probed_mask'])
	# bce = custom_bce(y['nsp_label'], x_cls)

	# acc = custom_acc(y['nsp_label'], x_cls)
	# r2 = custom_r2(y['magnitudes'], x_rec, y['probed_mask'])
	# print(r2)
	# break
# print(model.summary())

