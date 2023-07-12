import tensorflow as tf
import pandas as pd
import argparse
import os

from src.layers.downstream import AstromerEmbedding, ReduceAttention
from src.data.loaders import load_light_curves

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

from sklearn.metrics import precision_recall_fscore_support


def get_callbacks(path, patience=20, monitor='val_loss'):
	callbacks = [
		ModelCheckpoint(
			filepath=os.path.join(path, 'weights'),
			save_weights_only=True,
			monitor=monitor,
			save_best_only=True),
		EarlyStopping(monitor=monitor,
			patience = patience,
			restore_best_weights=True),
		TensorBoard(
			log_dir = os.path.join(path, 'logs'),
			histogram_freq=1,
			write_graph=True)]
	return callbacks


def create_classifier(pretrain_weights, window_size, n_classes, clf_name='mlp_att'):

	# Following the input definition of `format_input_lc()` at /src/data/loaders.py
	inp_placeholder = {
		'input': Input(shape=(window_size, 3), batch_size=None, name='values'),
		'mask'  : Input(shape=(window_size, ), batch_size=None, name='mask')
	}

	# Multilayer Perceptron + Attention Embedding
	if clf_name == 'mlp_att':
		_, x_rec = AstromerEmbedding(pretrain_weights=pretrain_weights, name='astromer')(inp_placeholder)
		x = ReduceAttention(reduce_to='mean',name='reduce_rec')(x_rec, inp_placeholder['mask'])
		x = Dense(1024, activation='relu')(x)
		x = Dense(512, activation='relu')(x)
		x = Dense(256, activation='relu')(x)

	# Just the [CLS] token
	if clf_name == 'mlp_cls':
		x, _ = AstromerEmbedding(pretrain_weights=pretrain_weights)(inp_placeholder)

	# [CLS] token concatenated with the average of the observation tokens
	if clf_name == 'mlp_all':
		x_cls, x_rec = AstromerEmbedding(pretrain_weights=pretrain_weights, name='astromer')(inp_placeholder)
		x = ReduceAttention(reduce_to='mean',name='reduce_rec')(x_rec, inp_placeholder['mask'])
		x = tf.concat([x_cls, x], axis=1, name='concat_cls_reduced')

	# Output layer
	x = LayerNormalization(name='layer_norm')(x)
	y_pred = Dense(n_classes, activation='relu', name='output_layer')(x)
	return CustomModel(inputs=inp_placeholder, outputs=y_pred, name=clf_name)

class CustomModel(Model):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

		self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
		self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
	
	def train_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			loss = self.compute_loss(y=y['label'], y_pred=y_pred)

		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		self.loss_tracker.update_state(loss)
		self.acc_metric.update_state(y['label'], y_pred)
		return {"loss": self.loss_tracker.result(), "acc": self.acc_metric.result()}
	@property
	def metrics(self):
		return [self.loss_tracker, self.acc_metric, self.val_loss_tracker, self.val_acc_metric]

	def test_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=False)  # Forward pass
			loss = self.compute_loss(y=y['label'], y_pred=y_pred)

		self.val_loss_tracker.update_state(loss)
		self.val_acc_metric.update_state(y['label'], y_pred)
		return {"loss": self.val_loss_tracker.result(), "acc": self.val_acc_metric.result()}

def run(opt):
	num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]
	train_batches = load_light_curves(os.path.join(opt.data, 'train'), 
									  num_cls=num_cls,
									  batch_size=opt.bs, 
									  window_size=opt.ws, 
									  repeat=1,
									  cache=True, 
									  njobs=None)
	valid_batches = load_light_curves(os.path.join(opt.data, 'val'), 
									  num_cls=num_cls,
									  batch_size=opt.bs, 
									  window_size=opt.ws, 
									  repeat=1,
									  cache=True, 
									  njobs=None)
	if opt.debug:
		train_batches = train_batches.take(1)
		valid_batches = valid_batches.take(1)
		opt.epochs = 10


	clf_model = create_classifier(opt.pre_weights, opt.ws, num_cls, clf_name=opt.clf_name)
	clf_model.compile(optimizer=Adam(opt.lr),
					  loss=CategoricalCrossentropy(from_logits=True))

	cbks = get_callbacks(opt.p)

	history = clf_model.fit(train_batches,
							epochs= opt.epochs,
							callbacks=cbks,
							validation_data=valid_batches)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pre-weights', default='./presentation/experiments/astromer_2/results/nsp_cond/1_4_64_rmse_0.5/pretraining', type=str,
					help='Pretrain weights folder')
	parser.add_argument('--p', default='./presentation/experiments/astromer_2/results/nsp_cond/1_4_64_rmse_0.5/classification/test', type=str,
					help='Project folder. Here will be stored weights and logs')
	parser.add_argument('--data', default='./data/records/atlas/fold_0/atlas_20', type=str,
					help='Data folder where tf.record files are located')
	parser.add_argument('--clf-name', default='mlp_all', type=str,
						help='Classifier name: mlp_att, mlp_cls, mlp_all')
	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


	parser.add_argument('--lr', default=1e-3, type=float,
						help='learning rate')
	parser.add_argument('--bs', default=32, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--epochs', default=100000, type=int,
						help='Number of epochs')
	parser.add_argument('--ws', default=200, type=int,
						help='windows size of the PSFs')


	opt = parser.parse_args()        
	run(opt)