'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import pandas as pd
import os, sys

from src.data import pretraining_pipeline
from tensorflow.keras.callbacks  import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

exp_path = './presentation/scripts/results'
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
pretrained_weigths = './weights/macho'
astromer.load_weights(pretrained_weigths)

#

# # Train ASTROMER
# _ = astromer.fit(data['train'],
#               epochs=config[step]['epochs'],
#               validation_data=data['val'],
#               callbacks=cbks)
