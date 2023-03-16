'''
HYPERPARAMETER TUNNING USING OPTUNA 
BY Cristobal 2023
'''
import tensorflow as tf
import optuna 
import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from src.training import CustomSchedule
from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline
from optuna.integration import TFKerasPruningCallback, OptunaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import CategoricalCrossentropy


def create_classifier(astromer, z_dim, num_cls, n_steps=200, name='mlp_att'):
	placeholder = build_input(n_steps)
	encoder = astromer.get_layer('encoder')
	encoder.trainable = False
	conv_shift = 4
	x = encoder(placeholder, training=False)
	x = Conv1D(32, 5, activation='relu', input_shape=[n_steps, z_dim])(x)
	x = Flatten()(tf.expand_dims(x, 1))
	x = tf.reshape(x, [-1, (n_steps-conv_shift)*32])
	x = Dense(num_cls)(x)
	return Model(inputs=placeholder, outputs=x, name=name)

def objetive(trial):

	n_layers 	 = trial.suggest_int('n_layers', 1, 8)
	n_heads  	 = trial.suggest_int('n_heads', 1, 8)
	head_dim 	 = trial.suggest_categorical('head_dim', [16, 32, 64, 128, 256])
	dff  		 = trial.suggest_int('dff', 32, 256)
	dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

	batch_size = 256 # Max batch size based on the heaviest model
	window_size = 200
	data_path = './data/records/macho'

	d_model = head_dim*n_heads
	astromer =  get_ASTROMER(num_layers=n_layers,
							 d_model=d_model,
							 num_heads=n_heads,
							 dff=dff,
							 base=1000,
							 dropout=dropout_rate,
							 maxlen=window_size,
							 pe_c=1.)
	lr = CustomSchedule(d_model)
	optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	astromer.compile(optimizer=optimizer)

	batch_size = (total_params * 661505)

	data = dict()
	for subset in ['train', 'val']:
		data[subset] = pretraining_pipeline(data_path, batch_size, window_size, .5, .2,	.2,
											sampling=True, shuffle=True, repeat=1, num_cls=None,
											normalize=True, cache=True)

	earlystop_callback   = EarlyStopping(monitor='val_loss', patience = 20, restore_best_weights=True),
	checkpoint_callback  = ModelCheckpoint(filepath=f'./results/hp/trial_{trial.number}/pretraining/model_weights.h5', save_best_only=True)
	tensorboard_callback = TensorBoard(log_dir=f'./results/hp/trial_{trial.number}/pretraining/logs')
	optuna_callback      = OptunaCallback(trial, metric_name='val_loss')
	pruning_callback     = TFKerasPruningCallback(trial, 'val_loss')

	astromer.fit(data['train'],
	 			 epochs=10000,
				 validation_data=data['val'], 
				 callbacks=[earlystop_callback, 
				 			checkpoint_callback, 
				 			tensorboard_callback, 
				 			optuna_callback,
				 			pruning_callback])
	# ==========================================================================================
	# ======= CLASSIFICATION ===================================================================
	# ==========================================================================================
	data = dict()
	fold_n = 1
	for subset in ['train', 'val']:
		num_cls = pd.read_csv(
			os.path.join(f'./data/records/alcock/fold_{fold_n}/alcock_50/', 'objects.csv')).shape[0]

		data[subset] = pretraining_pipeline(
			f'./data/records/alcock/fold_{fold_n}/alcock_50/{subset}',
			batch_size,	200, 0., 0., 0., False,	True, repeat=1,	num_cls=num_cls,
			normalize=True, cache=True)

		clf_model = create_classifier(astromer, d_model, num_cls, n_steps=window_size, name='mlp_att')
		optimizer = Adam(learning_rate=1e-3)
		clf_model.compile(optimizer=optimizer,
						  loss=CategoricalCrossentropy(from_logits=True),
						  metrics='accuracy')

		earlystop_callback   = EarlyStopping(monitor='val_loss', patience = 20, restore_best_weights=True),
		tensorboard_callback = TensorBoard(log_dir=f'./results/hp/trial_{trial.number}/classification')
		optuna_callback      = OptunaCallback(trial, metric_name='val_loss')
		pruning_callback     = TFKerasPruningCallback(trial, 'val_loss')

		_ = clf_model.fit(data['train'],
						  epochs=1000,
						  callbacks=[earlystop_callback,
									 tensorboard_callback,
									 optuna_callback],
						  validation_data=data['val'])

		loss, acc = clf_model.evaluate(data['val'])
		
		optuna.report(acc, step=trial.epoch)
		return acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"