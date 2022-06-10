#!/usr/bin/python
import pandas as pd
import subprocess
import os, sys
import sys
import time
import json

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, LSTM, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from core.data import load_dataset, inference_pipeline
from presentation.experiments.clf.classifiers import build_lstm, \
                                                     build_lstm_att, \
                                                     build_mlp_att
from core.astromer import ASTROMER


os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
ds_name = sys.argv[2]
case = sys.argv[3]

project_dir = './presentation/experiments/clf/finetuning_2/{}/'.format(ds_name)
data_path   = './data/records/{}/'.format(ds_name)
exp_name    = './presentation/experiments/clf/classifiers_2/{}/{}/'.format(case, ds_name)

max_obs = 200
batch_size = 512
print('BATCH_SIZE: ',batch_size)

datasets = ['{}_500'.format(ds_name),
            '{}_100'.format(ds_name),
            '{}_20'.format(ds_name),
            '{}_50'.format(ds_name),
            ]

if case == 'a':
    print('[INFO] No training ASTROMER')
    train_astromer = False
    models_arch = ['lstm', 'mlp_att', 'lstm_att']
else:
    train_astromer = True
    models_arch = ['mlp_att', 'lstm_att']
    
for model_arch in models_arch:
    for ds in datasets:
        for fold_n in range(3):
            astroweights = '{}/fold_{}/{}'.format(project_dir, fold_n, ds)
            ds_path = '{}/fold_{}/{}'.format(data_path, fold_n, ds)
            target_dir = '{}/fold_{}/{}/{}'.format(exp_name, fold_n, ds, model_arch)

            n_classes = pd.read_csv(os.path.join(ds_path, 'objects.csv')).shape[0]
            dataset = load_dataset(os.path.join(ds_path, 'train'),repeat=1)
            train_batches = inference_pipeline(dataset,
                                               batch_size=batch_size,
                                               max_obs=max_obs,
                                               n_classes=n_classes,
                                               shuffle=True)

            dataset = load_dataset(os.path.join(ds_path, 'val'),repeat=1)
            val_batches = inference_pipeline(dataset,
                                             batch_size=batch_size,
                                             max_obs=max_obs,
                                             n_classes=n_classes,
                                             shuffle=False)


            if model_arch == 'mlp_att':
                astromer = ASTROMER()
                astromer.load_weights(astroweights)
                model = build_mlp_att(astromer, max_obs, n_classes=n_classes,
                                      train_astromer=train_astromer)

            if model_arch == 'lstm_att':
                astromer = ASTROMER()
                astromer.load_weights(astroweights)
                model = build_lstm_att(astromer, max_obs, n_classes=n_classes,
                                       train_astromer=train_astromer)

            if model_arch == 'lstm':
                model = build_lstm(max_obs, n_classes=n_classes,
                                   state_dim=296)
                
            os.makedirs(target_dir, exist_ok=True)
            
            optim = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optim,
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics='accuracy')
            
            estop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=40,
                                  verbose=0,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=True)
            tb = TensorBoard(log_dir=os.path.join(target_dir, 'logs'),
                             write_graph=False,
                             write_images=False,
                             update_freq='epoch',
                             profile_batch=0,
                             embeddings_freq=0,
                             embeddings_metadata=None)

            _ = model.fit(train_batches,
                          epochs=10000,
                          callbacks=[estop, tb],
                          validation_data=val_batches)

            model.save(os.path.join(target_dir, 'weights'))
