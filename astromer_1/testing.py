import keras
import pandas as pd
import os, sys
import tensorflow as tf
from src.data import pretraining_pipeline, load_data
from src.models.astromer_1 import get_ASTROMER, train_step, test_step
from presentation.experiments.utils import train_classifier
from src.models.astromer_1 import get_ASTROMER, build_input, train_step, test_step
from src.training.utils import train
from src.layers.LoRALayer import LoraLayer
from src.data import pretraining_pipeline, load_data
import matplotlib.pyplot as plt
import math
import keras
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)

import time
start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

weights_path = './astromer_1/pretraining'
BATCH_SIZE = 2500
data_path = './records/ogle/fold_0/ogle'
exp_path = './results/testing'


astromer =  get_ASTROMER(num_layers=2,
                num_heads=4,
                head_dim=64,
                mixer_size=128,
                dropout=0.0,
                pe_base=1000,
                pe_dim=256,
                pe_c=2,
                window_size=200,
                batch_size=2500,
                encoder_mode='normal',
                average_layers=False,
                mask_format='first')


train_batches = load_data(dataset='{}/train'.format(data_path),
                                    batch_size=BATCH_SIZE,
                                    window_size=200,
                                    probed=0.2, 
                                    random_same=0.1,
                                    sampling=True,
                                    off_nsp=True, 
                                    repeat=4)
valid_batches = load_data(dataset='{}/val'.format(data_path),
                                    batch_size=BATCH_SIZE,
                                    window_size=200,
                                    off_nsp=True, 
                                    probed=0.2,
                                    random_same=0.1,
                                    sampling=True,
                                    repeat=1)
test_loader = load_data(dataset='{}/test'.format(data_path), 
                            batch_size=BATCH_SIZE, 
                            probed=0.2,  
                            random_same=0.1,
                            window_size=200, 
                            off_nsp=True, 
                            repeat=1, 
                            sampling=True)

astromer.load_weights(os.path.join(weights_path, 'weights', 'weights'))

astromer, (best_train_log, best_val_log, test_metrics) = train(astromer,
           train_batches, 
            valid_batches, 
            num_epochs=10000, 
            lr= 1e-5, 
            test_loader=test_loader,
            project_path=exp_path,
            debug=False,
            patience=20,
            train_step_fn=train_step,
            test_step_fn=test_step )

print(best_train_log, best_val_log, test_metrics)