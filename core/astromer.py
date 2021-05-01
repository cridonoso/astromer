import tensorflow as tf
from tqdm import tqdm
import os

from core.losses  import custom_mse, custom_bce
from core.output  import RegLayer, ClfLayer
from core.input   import input_format
from core.metrics import custom_acc
from core.encoder import Encoder
from core.decoder import Decoder

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 maxlen=100,
                 batch_size=None):

    serie = Input(shape=((maxlen*2)+3, 1),
                  batch_size=batch_size,
                  name='values')
    times = Input(shape=((maxlen*2)+3,1),
                  batch_size=batch_size,
                  name='times')
    mask  = Input(shape=(1, (maxlen*2)+3,(maxlen*2)+3),
                  batch_size=batch_size,
                  name='mask')
    placeholder = {'values':serie, 'mask':mask, 'times':times}

    x = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=10000,
                rate=0.1,
                name='encoder')(placeholder)

    x_reg = RegLayer(name='regression')(x)
    x_clf = ClfLayer(name='classification')(x)

    return Model(inputs=placeholder,
                 outputs=(x_reg, x_clf),
                 name="ASTROMER")

@tf.function
def train_step(model, batch, opt, num_cls=2):
    inputs, target = input_format(batch)

    x_true = tf.slice(target['x_true'], [0,0,1], [-1,-1,1])
    y_true = tf.slice(target['y_true'], [0,0,0], [-1,-1,1])

    with tf.GradientTape() as tape:
        x_pred, y_pred = model(inputs)

        mse = custom_mse(y_true=x_true,
                         y_pred=x_pred,
                         sample_weight=target['weigths'],
                         mask=target['x_mask'])

        bce = custom_bce(y_true=y_true,
                         y_pred=y_pred,
                         num_cls=num_cls)

        loss = bce + mse

        acc = custom_acc(y_true, y_pred)

    grads = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return loss, acc

@tf.function
def valid_step(model, batch, num_cls=2):
    inputs, target = input_format(batch)

    x_true = tf.slice(target['x_true'], [0,0,1], [-1,-1,1])
    y_true = tf.slice(target['y_true'], [0,0,0], [-1,-1,1])

    with tf.GradientTape() as tape:
        x_pred, y_pred = model(inputs)

        mse = custom_mse(y_true=x_true,
                         y_pred=x_pred,
                         sample_weight=target['weigths'],
                         mask=target['x_mask'])

        bce = custom_bce(y_true=y_true,
                         y_pred=y_pred,
                         num_cls = num_cls)

        loss = bce + mse
        acc = custom_acc(y_true, y_pred)
    return loss, acc


def train(model,
          train_dataset,
          valid_dataset,
          num_cls=2,
          patience=20,
          exp_path='./experiments/test',
          epochs=1):

    os.makedirs(exp_path, exist_ok=True)
    # Optimizer
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    # To save metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # Training Loop
    best_loss = 999999.
    es_count = 0
    for epoch in range(epochs):
        for step, train_batch in tqdm(enumerate(train_dataset), desc='train'):
            loss, acc = train_step(model, train_batch, optimizer,
                                   num_cls=num_cls)
            train_loss.update_state(loss)
            train_acc.update_state(acc)

        for valid_batch in tqdm(valid_dataset, desc='validation'):
            loss, acc = valid_step(model, valid_batch,
                                   num_cls=num_cls)
            valid_loss.update_state(loss)
            valid_acc.update_state(acc)


        if valid_loss.result() < best_loss:
            best_loss = valid_loss.result()
            print(best_loss)
            es_count = 0.
            model.save_weights(exp_path+'/weights')
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break

        valid_loss.reset_states()
        train_loss.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()


def get_FINETUNING(astromer, num_cls=2):
    encoder = astromer.get_layer('encoder')
    x_reg = astromer.get_layer('regression')(encoder.output)
    x_clf = ClfLayer(num_cls=num_cls, name='NewClf')(encoder.output)

    return Model(inputs=encoder.input,
                 outputs=[x_reg, x_clf],
                 name="FINETUNING")
