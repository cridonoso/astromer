import tensorflow as tf
from tqdm import tqdm
import os, sys

from core.output  import RegLayer, ClfLayer, SplitLayer
from core.tboard  import save_scalar, draw_graph
from core.losses  import custom_mse, custom_bce
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
                 base=10000,
                 dropout=0.1,
                 maxlen=100,
                 batch_size=None):

    serie = Input(shape=(maxlen+3, 1),
                  batch_size=batch_size,
                  name='input')
    times = Input(shape=(maxlen+3, 1),
                  batch_size=batch_size,
                  name='times')
    mask  = Input(shape=(maxlen+3, 1),
                  batch_size=batch_size,
                  name='mask')
    segsep = Input(shape=(),
                  batch_size=batch_size,
                  name='segsep')
    placeholder = {'input':serie, 'mask':mask, 'times':times, 'segsep':segsep}

    x = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=base,
                rate=dropout,
                name='encoder')(placeholder)
    x_cls, \
    x_reg = SplitLayer(name='split_z')(x)
    x_reg = RegLayer(name='regression')(x_reg)
    x_cls = ClfLayer(name='classification')(x_cls)

    return Model(inputs=placeholder,
                 outputs=(x_reg, x_cls),
                 name="ASTROMER")

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        x_pred, y_pred = model(batch)
        mse = custom_mse(y_true=batch['input'],
                         y_pred=x_pred,
                         mask=batch['mask'])

        bce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        loss = bce + mse
        acc = custom_acc(batch['label'], y_pred)

    grads = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return loss, acc, bce, mse

@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        x_pred, y_pred = model(batch)

        mse = custom_mse(y_true=batch['input'],
                         y_pred=x_pred,
                         mask=batch['mask'])
        bce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)

        loss = bce + mse
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return loss, acc, bce, mse, x_pred, y_pred, batch['input'], batch['label']
    return loss, acc, bce, mse

def train(model,
          train_dataset,
          valid_dataset,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          finetuning=False,
          use_random=True,
          num_cls=2,
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
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    # To save metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_mse = tf.keras.metrics.Mean(name='train_mse')
    valid_mse = tf.keras.metrics.Mean(name='valid_mse')
    train_bce = tf.keras.metrics.Mean(name='train_bce')
    valid_bce = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # Training Loop
    best_loss = 999999.
    es_count = 0
    for epoch in range(epochs):
        for step, train_batch in tqdm(enumerate(train_dataset), desc='train'):
            loss, acc, bce, mse = train_step(model, train_batch, optimizer)
            train_loss.update_state(loss)
            train_acc.update_state(acc)
            train_bce.update_state(bce)
            train_mse.update_state(mse)

        for valid_batch in tqdm(valid_dataset, desc='validation'):
            loss, acc, bce, mse = valid_step(model, valid_batch)
            valid_loss.update_state(loss)
            valid_acc.update_state(acc)
            valid_bce.update_state(bce)
            valid_mse.update_state(mse)

        save_scalar(train_writter, train_loss, epoch, name='loss')
        save_scalar(valid_writter, valid_loss, epoch, name='loss')
        save_scalar(train_writter, train_acc, epoch, name='accuracy')
        save_scalar(valid_writter, valid_acc, epoch, name='accuracy')
        save_scalar(train_writter, train_mse, epoch, name='mse')
        save_scalar(valid_writter, valid_mse, epoch, name='mse')
        save_scalar(train_writter, train_bce, epoch, name='bce')
        save_scalar(valid_writter, valid_bce, epoch, name='bce')

        if verbose == 0:
            print('EPOCH {} - ES COUNT: {}'.format(epoch, es_count))
            print('train loss: {:.2f} - train acc: {:.2f} - train ce: {:.2f}, train mse: {:.4f}'.format(train_loss.result(),
                                                                                                        train_acc.result(),
                                                                                                        train_bce.result(),
                                                                                                        train_mse.result(),
                                                                                                        ))
            print('val loss: {:.2f} - val acc: {:.2f} - val ce: {:.2f}, val mse: {:.4f}'.format(valid_loss.result(),
                                                                                                        valid_acc.result(),
                                                                                                        valid_bce.result(),
                                                                                                        valid_mse.result(),
                                                                                                        ))
        if valid_loss.result() < best_loss:
            best_loss = valid_loss.result()
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

        train_bce.reset_states()
        valid_bce.reset_states()

        train_mse.reset_states()
        valid_mse.reset_states()

def predict(model,
            dataset,
            conf,
            predic_proba=False):
    preds, reconstructions = [], []
    true_cls, true_x = [],[]
    total_loss, total_acc, total_bce, total_mse = [], [], [], []
    for step, batch in tqdm(enumerate(dataset), desc='prediction'):
        loss, acc, bce, mse, \
        x_pred, y_pred, \
        x_true, y_true = valid_step(model, batch,
                                    return_pred=True)
        total_loss.append(loss)
        total_acc.append(acc)
        total_bce.append(bce)
        total_mse.append(mse)
        true_cls.append(tf.squeeze(y_true))
        true_x.append(x_true)
        preds.append(tf.squeeze(y_pred))
        reconstructions.append(x_pred)

    y_pred = tf.concat(preds, 0)
    if not predic_proba:
        y_pred = tf.argmax(y_pred, 1)

    res = {'loss':tf.reduce_mean(total_loss).numpy(),
           'acc':tf.reduce_mean(total_acc).numpy(),
           'bce':tf.reduce_mean(total_bce).numpy(),
           'mse':tf.reduce_mean(total_mse).numpy(),
           'y_pred':y_pred,
           'x_pred': tf.concat(reconstructions, 0),
           'y_true':tf.concat(true_cls, 0),
           'x_true': tf.concat(true_x, 0)}
    return res
