'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import pandas as pd
import os, sys

from src.data import pretraining_pipeline
from src.models import get_ASTROMER

from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = '-1' if len(sys.argv) == 1 else sys.argv[1]

exp_path = './presentation/scripts/results/test'
head_dim = 64
num_head = 4
d_model  = head_dim * num_head

# Creating ASTROMER
astromer =  get_ASTROMER(num_layers=2,
                         d_model=d_model,
                         num_heads=num_head,
                         dff=128,
                         base=1000,
                         dropout=0.,
                         maxlen=200,
                         no_train=False)
optimizer = Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
astromer.compile(optimizer=optimizer)

# LOADING DATA
BATCH_SIZE = 2000
train_batches = pretraining_pipeline('./data/records/macho/train',
                                     batch_size=BATCH_SIZE,
                                     window_size=200,
                                     msk_frac=.5,
                                     rnd_frac=.2,
                                     same_frac=.2,
                                     sampling=True,
                                     shuffle=True,
                                     repeat=4,
                                     normalize=True,
                                     cache=True)
valid_batches = pretraining_pipeline('./data/records/new_ztf_g/val',
                                     batch_size=BATCH_SIZE,
                                     window_size=200,
                                     msk_frac=.5,
                                     rnd_frac=.2,
                                     same_frac=.2,
                                     sampling=True,
                                     shuffle=False,
                                     repeat=1,
                                     normalize=True,
                                     cache=True)

# train_batches = train_batches.take(5) # for testing purposes only
# valid_batches = valid_batches.take(5) # for testing purposes only

# CALLBACKS
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(exp_path, 'weights'),
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True),
    EarlyStopping(monitor='val_loss',
        patience = 20,
        restore_best_weights=True),
    TensorBoard(
        log_dir = os.path.join(exp_path, 'logs'),
        histogram_freq=1,
        write_graph=False)]

# TRAIN ASTROMER
_ = astromer.fit(train_batches,
                 epochs=1000,
                 validation_data=valid_batches,
                 callbacks=callbacks)
