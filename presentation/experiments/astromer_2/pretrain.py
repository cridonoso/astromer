import tensorflow as tf 
import pandas as pd
import json
import sys
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)
from src.models import get_ASTROMER_II
from src.data import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# BEST PARAMETERS ACCORDING TO PREVIOUS EXPERIMENTS
datapath      = './data/records/macho_clean'

n_layers      = int(sys.argv[2])
n_heads       = 2
head_dim      = 64
mixer_size    = 64
learning_rate = 1e-5
dropout_rate  = 0.2
window_size   = 200
batch_size    = 256
probed        = 0.6
rand          = 0.2
nsp_prob      = 0.5

MASTER_PROJECT_NAME = 'nsp_script_0dp'
ROOT = './presentation/experiments/astromer_2/'
EXPDIR = os.path.join(ROOT, 'results', MASTER_PROJECT_NAME)
os.makedirs(EXPDIR, exist_ok=True)


# ========== DATA ========================================
train_batches = load_data(dataset=os.path.join(datapath, 'train'), 
                          batch_size=batch_size, 
                          probed=probed,  
                          window_size=window_size, 
                          nsp_prob=nsp_prob, 
                          repeat=1, 
                          sampling=True)
valid_batches = load_data(dataset=os.path.join(datapath, 'val'), 
                          batch_size=batch_size, 
                          probed=probed,  
                          window_size=window_size, 
                          nsp_prob=nsp_prob, 
                          repeat=1, 
                          sampling=True)

# ======= MODEL ========================================
model_name = '{}_{}_{}'.format(n_layers, n_heads, head_dim)
PTWEIGTHS = os.path.join(EXPDIR, model_name, 'pretraining')
        
d_model = head_dim*n_heads
astromer = get_ASTROMER_II(num_layers=n_layers,
                           num_heads=n_heads,
                           head_dim=head_dim,
                           mixer_size=mixer_size,
                           dropout=dropout_rate,
                           pe_base=1000,
                           pe_dim=128,
                           pe_c=1,
                           window_size=window_size)

optimizer = Adam(learning_rate)
astromer.compile(optimizer=optimizer)

callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(PTWEIGTHS, 'weights'),
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True),
        EarlyStopping(monitor='val_loss',
            patience = 20,
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(PTWEIGTHS, 'logs'),
            histogram_freq=1,
            write_graph=True)]
    
astromer.fit(train_batches, 
         epochs=100000, 
         validation_data=valid_batches,
         callbacks=callbacks)      

# ======== TESTING =========================================
test_batches = load_data(dataset=os.path.join(datapath, 'test'), 
                          batch_size=batch_size, 
                          probed=probed,  
                          window_size=window_size, 
                          nsp_prob=nsp_prob, 
                          repeat=1, 
                          sampling=True)

acc, bce, loss, r2, rmse = astromer.evaluate(test_batches)   
with open(os.path.join(PTWEIGTHS, 'results.json'), 'w') as fp:
    json.dump({'test_acc': acc, 
               'test_r2':r2, 
               'test_rmse':rmse, 
               'test_bce':bce}, fp)


