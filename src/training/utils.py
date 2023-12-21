import tensorflow as tf
import toml
import time
import os

from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.optimizers import Adam
from src.training.scheduler import CustomSchedule
from tensorflow.keras.optimizers.experimental import AdamW
from tqdm import tqdm

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

def merge_metrics(**kwargs):
    merged = {}
    for key, value in kwargs.items():
        for subkey, subvalue in value.items():
            if key ==  'train':
                merged['{}'.format(subkey)] = subvalue
            else:
                merged['{}_{}'.format(key, subkey)] = subvalue
    return merged

def train(model, 
          train_loader, 
          valid_loader, 
          num_epochs=1000, 
          lr=1e-3, 
          project_path=None, 
          debug=False, 
          patience=20,
          train_step_fn=None,
          test_step_fn=None,
          argparse_dict=None,
          scheduler=False,
          callbacks=None,
          rmse_factor=0.5,
          reset_states=False):

    start = time.time()

    os.makedirs(project_path, exist_ok=True)

    print('[INFO] Project Path: {}'.format(os.path.join(project_path)))
    if debug:
        print('[INFO] DEBGUGING MODE')
        num_epochs   = 2
        train_loader = train_loader.take(5)
        valid_loader = valid_loader.take(5)

    if argparse_dict is not None:
        with open(os.path.join(project_path, 'config.toml'), 'w') as f:
            toml.dump(argparse_dict, f)

    __callbacks = CallbackList(callbacks=callbacks, model=model)
    if callbacks is not None: print('[INFO] Callbacks added')

    # ======= TRAINING LOOP =========
    if scheduler:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(argparse_dict['head_dim'])
    if reset_states: print('[INFO] Reset state activated')
    optimizer = Adam(lr, 
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9,
                     name='astromer_optimizer')
    es_count = 0
    min_loss = 1e9
    best_train_log, best_val_log = None, None
    ebar = tqdm(range(num_epochs), total=num_epochs)
    logs = {}
    __callbacks.on_train_begin(logs=logs)
    for epoch in ebar:
        __callbacks.on_epoch_begin(epoch, logs=logs)
        train_logs, valid_logs = [], [] 

        for batch, (x, y) in enumerate(train_loader):
            __callbacks.on_batch_begin(batch, logs=logs)
            __callbacks.on_train_batch_begin(batch, logs=logs)
            logs = train_step_fn(model, x, y, optimizer, rmse_factor=rmse_factor)
            train_logs.append(logs)
            __callbacks.on_train_batch_end(batch, logs=logs)
            __callbacks.on_batch_end(batch, logs=logs)
        
        if reset_states: model.reset_states()

        for x, y in valid_loader:
            __callbacks.on_batch_begin(batch, logs=logs)
            __callbacks.on_test_batch_begin(batch, logs=logs)
            logs = test_step_fn(model, x, y, rmse_factor=rmse_factor)
            valid_logs.append(logs)
            __callbacks.on_test_batch_end(batch, logs=logs)
            __callbacks.on_batch_end(batch, logs=logs)

        epoch_train_metrics = average_logs(train_logs)
        epoch_valid_metrics = average_logs(valid_logs)
        logs = merge_metrics(train=epoch_train_metrics, val=epoch_valid_metrics)
        __callbacks.on_epoch_end(epoch, logs=logs)
        
        ebar.set_description('STOP: {:02d}/{:02d} LOSS: {:.3f}/{:.3f} R2:{:.3f}/{:.3f}'.format(es_count, 
                                                                            patience, 
                                                                            logs['loss'],
                                                                            logs['val_loss'],
                                                                            logs['r_square'],
                                                                            logs['val_r_square']))

    elapsed = time.time() - start
    __callbacks.on_train_end(logs=logs)

    return model