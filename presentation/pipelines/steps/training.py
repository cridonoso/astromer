import tensorflow as tf
import toml
import os

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.training.scheduler import CustomSchedule

def finetune_model(model, data, config):
        
    ft_data_info = config['ft_data'].split('./data/records')[-1]
    print(ft_data_info)
    EXPDIR = os.path.join(config['pt_path'], '..', 'finetuning', *ft_data_info.split('/'))
    os.makedirs(EXPDIR, exist_ok=True)

    if config['scheduler']:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(d_model=int(config['head_dim']*config['num_heads']))
    else:
        lr = config['lr']

    model.compile(optimizer=Adam(lr, 
                  beta_1=0.9,
                  beta_2=0.98,
                  epsilon=1e-9,
                  name='astromer_optimizer'))

    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(config, f)

    cbks = [TensorBoard(log_dir=os.path.join(EXPDIR, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=25),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1)]

    model.fit(data['train'], 
              epochs= 1000, 
              validation_data=data['validation'],
              callbacks=cbks)