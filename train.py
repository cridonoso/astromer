import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

from core.data import make_batches, positional_encoding, tokenizers
from core.transformer import Transformer, MiniTransformer
from core.scheduler import CustomSchedule
from core.callbacks import get_callbacks

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# Hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
exp_path = "./experiments/train"
EPOCHS = 20
BATCHSIZE = 64

# Loading data
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# Create batches: tokenize and encode
train_batches = make_batches(train_examples, batchsize=BATCHSIZE)
val_batches   = make_batches(val_examples, batchsize=BATCHSIZE)

# Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

# Loss Function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                            reduction='none')
# Metrics
transformer = MiniTransformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              input_vocab_size=tokenizers.pt.get_vocab_size(),
                              target_vocab_size=tokenizers.en.get_vocab_size(),
                              pe_input=1000,
                              rate=0.1)

transformer.model(BATCHSIZE).summary()

transformer.compile(optimizer=optimizer, 
                    loss_function=loss_object,
                    metrics=['accuracy', loss_object])

transformer.fit(train_batches.take(5), 
                epochs=EPOCHS, 
                verbose=1, 
                validation_data=val_batches.take(5),
                callbacks=get_callbacks(exp_path))

