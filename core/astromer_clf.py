import tensorflow as tf
from tqdm import tqdm
import os

from core.output  import ClfLayer
from core.tboard  import save_scalar, draw_graph
from core.losses  import custom_mse, custom_bce
from core.metrics import custom_acc
from core.encoder import Encoder
from core.decoder import Decoder

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def get_CLASSIFIER(astromer, units, dropout=0.25, num_cls=2):
    encoder = astromer.get_layer('encoder')
    x_rnn = LSTM(units, dropout=dropout, return_sequences=True)(encoder.output)
    x_rnn = LSTM(units, dropout=dropout)(x_rnn)
    x_cls = ClfLayer(name='classification', num_cls=num_cls)(x_rnn)
    return Model(inputs=encoder.input,
                 outputs=x_cls,
                 name="Classifier")

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        y_pred = model(batch)
        ce = custom_bce(y_true=batch['label'], y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    grads = tape.gradient(ce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return acc, ce

@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        y_pred = model(batch)
        ce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return acc, ce, x_pred, y_pred, batch['input'], batch['label']
    return acc, ce

def train(model,
          train_dataset,
          valid_dataset,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          verbose=1):

    os.makedirs(exp_path, exist_ok=True)

    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'valid'))

    batch = [t for t in train_dataset.take(1)][0]
    draw_graph(model, batch, train_writter, exp_path)

    # Optimizer
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    # To save metrics
    train_bce  = tf.keras.metrics.Mean(name='train_bce')
    valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # Training Loop
    best_loss = 999999.
    es_count = 0
    for epoch in range(epochs):
        for step, train_batch in tqdm(enumerate(train_dataset), desc='train'):

            acc, bce = train_step(model, train_batch, optimizer)
            train_acc.update_state(acc)
            train_bce.update_state(bce)

        for valid_batch in tqdm(valid_dataset, desc='validation'):
            acc, bce = valid_step(model, valid_batch)
            valid_acc.update_state(acc)
            valid_bce.update_state(bce)

        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_bce, epoch, name='xentropy')
        save_scalar(valid_writter, valid_bce, epoch, name='xentropy')

        if verbose == 0:
            print('EPOCH {} - ES COUNT: {}'.format(epoch, es_count))
            print('train acc: {:.2f} - train ce: {:.2f}'.format(train_acc.result(),
                                                                train_bce.result()))
            print('val acc: {:.2f} - val ce: {:.2f}'.format(valid_acc.result(),
                                                            valid_bce.result(),
                                                            ))
        if valid_bce.result() < best_loss:
            best_loss = valid_bce.result()
            es_count = 0.
            model.save_weights(exp_path+'/weights')
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break

        valid_bce.reset_states()
        train_bce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()
