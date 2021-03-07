import tensorflow as tf
import logging
import time
import os

from core.data  import load_records
from core.transformer import ASTROMER
from core.scheduler import CustomSchedule
from core.callbacks import get_callbacks
from core.losses import CustomMSE, ASTROMERLoss
from core.metrics import CustomACC

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# Hyperparameters
num_layers = 2 # stacked encoders 
d_model = 256 # Head-attention Dimensionality 
dff = 2048 # Middle Dense Layer Dimensionality
num_heads = 4
dropout_rate = 0.1
exp_path = "./experiments/train"
root_data = './data/records/macho'
EPOCHS = 2
BATCHSIZE = 16

# Loading data
train_batches = load_records(os.path.join(root_data, 'train'), BATCHSIZE)
valid_batches = load_records(os.path.join(root_data, 'val'), BATCHSIZE)
test_batches = load_records(os.path.join(root_data, 'test'), BATCHSIZE)

# Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
# Model Instance
transformer = ASTROMER(num_layers=num_layers,
                       d_model=d_model,
                       num_heads=num_heads,
                       dff=dff,
                       pe_input=1000,
                       rate=0.1)
# Compile
transformer.compile(optimizer=optimizer, 
                    loss=ASTROMERLoss(),
                    metrics=[CustomMSE(), CustomACC()])
# Create graph
transformer.model(BATCHSIZE).summary()
# Training
transformer.fit(train_batches, 
                epochs=EPOCHS, 
                verbose=1,
                validation_data=valid_batches,
                callbacks=get_callbacks())
# Testing
transformer.evaluate(test_batches)