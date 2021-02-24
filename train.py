import tensorflow as tf
import logging
import time

from core.data  import load_records
from core.transformer import ASTROMER
from core.scheduler import CustomSchedule
from core.callbacks import get_callbacks

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# Hyperparameters
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.1
exp_path = "./experiments/train"
root_data = './data/records/macho'
EPOCHS = 20
BATCHSIZE = 16

# Loading data
train_batches, val_batches = load_records(root_data, BATCHSIZE)



# Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

# Loss Function
cls_loss_object = BinaryCrossentropy(from_logits=True, reduction='none')
rec_loss_object = MeanSquaredError(reduction='none')

# Metrics
transformer = ASTROMER(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              pe_input=1000,
                              rate=0.1)

transformer.model(BATCHSIZE).summary()


transformer.compile(optimizer=optimizer, 
                    cls_loss=cls_loss_object,
                    rec_loss=rec_loss_object,
                    metrics=[rec_loss_object])

transformer.fit(train_batches, 
                epochs=EPOCHS, 
                verbose=1,
                validation_data=val_batches,
                callbacks=get_callbacks())

# TODO Create custom metric