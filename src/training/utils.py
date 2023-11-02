import pandas as pd
import tensorflow as tf
import toml
import yaml
import time
import os

from tensorflow.keras.optimizers import Adam
from src.training.scheduler import CustomSchedule
from tqdm import tqdm

from tensorflow.keras import backend as K

def tensorboard_log(logs, writer, step=0):
	with writer.as_default():
		for key, value in logs.items():
			tf.summary.scalar(key, value, step=step)

def average_logs(logs):
	N = len(logs)
	average_dict = {}
	for key in logs[0].keys():
		sum_log = sum(log[key] for log in logs)
		average_dict[key] = float(sum_log/N)
	return average_dict

def train(model, 
		  train_loader, 
		  valid_loader, 
		  num_epochs=1000, 
		  lr=1e-3, 
		  test_loader=None, 
		  project_path=None, 
		  debug=False, 
		  patience=20,
		  train_step_fn=None,
		  test_step_fn=None,
		  argparse_dict=None):

	os.makedirs(project_path, exist_ok=True)

	if debug:
		print('[INFO] DEBGUGING MODE')
		num_epochs   = 2
		train_loader = train_loader.take(2)
		valid_loader = valid_loader.take(2)
		if test_loader is not None:
			test_loader = test_loader.take(1)

	if argparse_dict is not None:
		file = open(os.path.join(project_path, 'config.yaml'), "w")
		yaml.dump(argparse_dict, file)
		file.close()

		#with open(os.path.join(project_path, 'config.toml'), 'w') as f:
		#	toml.dump(argparse_dict, f)

	# ======= TRAINING LOOP =========
	train_writer = tf.summary.create_file_writer(os.path.join(project_path, 'logs', 'train'))
	valid_writer = tf.summary.create_file_writer(os.path.join(project_path, 'logs', 'validation'))

	print('[INFO] Logs: {}'.format(os.path.join(project_path, 'logs')))

	# Optimizer
	if lr == 'scheduler':
		print('[INFO] Using Custom Scheduler')
		lr = CustomSchedule(model.get_layer('encoder').head_dim)
	else:
		lr = float(lr)

	optimizer = Adam(lr, 
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9,
                     name='astromer_optimizer')

	print(f'optimizer.learning_rate: {optimizer.learning_rate}')

	# Training Loop
	es_count = 0
	min_loss = 1e9
	best_train_log, best_val_log = None, None
	ebar = tqdm(range(num_epochs), total=num_epochs)

	dict_epoch = {
        'epoch': [],
        'time_epoch': [],
	}

	dict_batch = {
		'batch': [],
		'time_batch': []
	}

	for epoch in ebar:

		epoch_start = time.time()

		train_logs, valid_logs = [], [] 
		for batch, (x, y) in enumerate(train_loader):
			start = time.time()
			logs = train_step_fn(model, x, y, optimizer)
			end = time.time()      

			train_logs.append(logs)
			dict_batch['batch'].append(batch)
			dict_batch['time_batch'].append(end - start)

		for x, y in valid_loader:
			logs = test_step_fn(model, x, y)
			valid_logs.append(logs)

		epoch_end = time.time()
		dict_epoch['epoch'].append(epoch)
		dict_epoch['time_epoch'].append(epoch_end - epoch_start)

		epoch_train_metrics = average_logs(train_logs)
		epoch_valid_metrics = average_logs(valid_logs)

		tensorboard_log(epoch_train_metrics, train_writer, step=epoch)
		tensorboard_log(epoch_valid_metrics, valid_writer, step=epoch)

		if tf.math.greater(min_loss, epoch_valid_metrics['loss']):
			min_loss = epoch_valid_metrics['loss']
			best_train_log = epoch_train_metrics
			best_val_log = epoch_valid_metrics
			es_count = 0
			model.save_weights(os.path.join(project_path, 'weights', 'weights'))
		else:
			es_count = es_count + 1

		if es_count == patience:
			print('[INFO] Early Stopping Triggered at epoch {:03d}'.format(epoch))
			break

		ebar.set_description('STOP: {:02d}/{:02d} LOSS: {:.3f}/{:.3f} R2:{:.3f}/{:.3f}'.format(es_count, 
																							   patience, 
																							   epoch_train_metrics['loss'],
																							   epoch_valid_metrics['loss'],
																							   epoch_train_metrics['r_square'],
																							   epoch_valid_metrics['r_square']))

	#if test_loader is not None:
	#	#K.clear_session()
	#	#if 'Finetuning' in argparse_dict.keys():
	#	#	print('Existeee fientuninggggg')
	#	#	best_model, _ = restore_ft_model(project_path, argparse_dict['Finetuning']['dataset'])
	#	#else:
	#	#	best_model, _ = restore_model(project_path, argparse_dict['dataset'])
	#	#best_model.load_weights(os.path.join(project_path, 'weights', 'weights'))
#
	#	#test_writer = tf.summary.create_file_writer(os.path.join(project_path, 'logs', 'test'))
	#	test_logs = []
	#	for x, y in test_loader:
	#		logs = test_step_fn(model, x, y)
	#		test_logs.append(logs)
	#	test_metrics = average_logs(test_logs)
	#	tensorboard_log(test_metrics, test_writer, step=0)
	#	tensorboard_log(test_metrics, test_writer, step=num_epochs)

	# Guarda el tiempo por epoch y batch
	df_time_batch = pd.DataFrame(data=dict_batch)
	df_time_batch.to_csv('{}/time_batch.csv'.format(project_path), index=False)

	df_time_epoch = pd.DataFrame(data=dict_epoch)
	df_time_epoch.to_csv('{}/time_epoch.csv'.format(project_path), index=False)

	return model, (best_train_log, best_val_log)