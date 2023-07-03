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
from src.data import pretraining_pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# BEST PARAMETERS ACCORDING TO PREVIOUS EXPERIMENTS
datapath      = './data/records/macho_clean'

n_layers      = int(sys.argv[2])
n_heads       = 4
head_dim      = 64
dff           = 64
learning_rate = 1e-5
dropout_rate  = 0.
window_size   = 200
batch_size    = 3000
probed        = 0.6
rand          = 0.2
nsp_prob      = 0.5
nsp_fraction  = 0.5

MASTER_PROJECT_NAME = 'nsp_script_0dp'
ROOT = './presentation/experiments/astromer_2/'
EXPDIR = os.path.join(ROOT, 'results', MASTER_PROJECT_NAME)
os.makedirs(EXPDIR, exist_ok=True)


# ========== DATA ========================================
train_batches = pretraining_pipeline(os.path.join(datapath, 'train'),
                                    batch_size,
                                    window_size,
                                    probed,
                                    rand,
                                    rand,
                                    True,
                                    True,
                                    repeat=4,
                                    num_cls=None,
                                    normalize='zero-mean',
                                    cache=True,
                                    nsp_prob=nsp_prob,
                                    nsp_frac=nsp_fraction)
valid_batches = pretraining_pipeline(os.path.join(datapath, 'val'),
                                    batch_size,
                                    window_size,
                                    probed,
                                    rand,
                                    rand,
                                    True,
                                    True,
                                    repeat=1,
                                    num_cls=None,
                                    normalize='zero-mean',
                                    cache=True,
                                    nsp_prob=nsp_prob,
                                    nsp_frac=nsp_fraction)

# ======= MODEL ========================================
model_name = '{}_{}_{}'.format(n_layers, n_heads, head_dim)
PTWEIGTHS = os.path.join(EXPDIR, model_name, 'pretraining')
        
d_model = head_dim*n_heads
astromer =  get_ASTROMER_II(num_layers=n_layers,
                            d_model=d_model,
                            num_heads=n_heads,
                            dff=dff,
                            base=10000,
                            dropout=dropout_rate,
                            maxlen=window_size,
                            pe_c=2)          

optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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
test_batches = pretraining_pipeline(os.path.join(datapath, 'test'),
                                    batch_size,
                                    window_size,
                                    probed,
                                    rand,
                                    rand,
                                    True,
                                    True,
                                    repeat=1,
                                    num_cls=None,
                                    normalize='zero-mean',
                                    cache=True,
                                    nsp_prob=nsp_prob,
                                    nsp_frac=nsp_fraction)

acc, bce, loss, r2, rmse = astromer.evaluate(test_batches)   
with open(os.path.join(PTWEIGTHS, 'results.json'), 'w') as fp:
    json.dump({'test_acc': acc, 
               'test_r2':r2, 
               'test_rmse':rmse, 
               'test_bce':bce}, fp)


