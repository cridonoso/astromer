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
                 base=10000,
                 dropout=0.1,
                 maxlen=100,
                 batch_size=None):

    serie = Input(shape=(maxlen+3, 1),
                  batch_size=batch_size,
                  name='values')
    times = Input(shape=(maxlen+3,1),
                  batch_size=batch_size,
                  name='times')
    mask  = Input(shape=(1, maxlen+3,maxlen+3),
                  batch_size=batch_size,
                  name='mask')
    placeholder = {'values':serie, 'mask':mask, 'times':times}

    x = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=base,
                rate=dropout,
                name='encoder')(placeholder)

    x_reg = RegLayer(name='regression')(x)
    x_clf = ClfLayer(name='classification')(x)

    return Model(inputs=placeholder,
                 outputs=(x_reg, x_clf),
                 name="ASTROMER")

@tf.function
def train_step(model, batch, opt, num_cls=2, use_random=True, finetuning=False):
    inputs, target = input_format(batch, use_random=use_random, finetuning=finetuning)

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
    return loss, acc, bce, mse

@tf.function
def valid_step(model, batch, num_cls=2, return_pred=False, use_random=True, finetuning=False):
    inputs, target = input_format(batch, use_random=use_random, finetuning=finetuning)

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
    if return_pred:
        return loss, acc, bce, mse, x_pred, y_pred, x_true, y_true
    return loss, acc, bce, mse

def save_scalar(writer, value, step, name=''):
    with writer.as_default():
        tf.summary.scalar(name, value.result(), step=step)

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

            loss, acc, bce, mse = train_step(model, train_batch, optimizer,
                                   num_cls=num_cls,
                                   use_random=use_random,
                                   finetuning=finetuning)
            train_loss.update_state(loss)
            train_acc.update_state(acc)
            train_bce.update_state(bce)
            train_mse.update_state(mse)

        for valid_batch in tqdm(valid_dataset, desc='validation'):
            loss, acc, bce, mse = valid_step(model, valid_batch,
                                   num_cls=num_cls,
                                   use_random=use_random,
                                   finetuning=finetuning)
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
            print('train loss: {:.2f} - train acc: {:.2f} - train ce: {:.2f}, train mse: {:.2f}'.format(train_loss.result(),
                                                                                                        train_acc.result(),
                                                                                                        train_bce.result(),
                                                                                                        train_mse.result(),
                                                                                                        ))
            print('train loss: {:.2f} - train acc: {:.2f} - train ce: {:.2f}, train mse: {:.2f}'.format(valid_loss.result(),
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

def predict(model, dataset, conf, predic_proba=False, use_random=True, finetuning=False):
    preds, reconstructions = [], []
    true_cls, true_x = [],[]
    total_loss, total_acc, total_bce, total_mse = [], [], [], []
    for step, batch in tqdm(enumerate(dataset), desc='prediction'):
        num_cls = batch['num_cls'][0]
        loss, acc, bce, mse, \
        x_pred, y_pred, \
        x_true, y_true = valid_step(model, batch, num_cls,
                                    return_pred=True,
                                    use_random=use_random,
                                    finetuning=finetuning)
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

def get_FINETUNING(astromer, num_cls=2):
    encoder = astromer.get_layer('encoder')
    x_reg = astromer.get_layer('regression')(encoder.output)
    x_clf = ClfLayer(num_cls=num_cls, name='NewClf')(encoder.output)

    return Model(inputs=encoder.input,
                 outputs=[x_reg, x_clf],
                 name="FINETUNING")

def get_attention(model, num_cls=2):
    encoder = model.get_layer('encoder')
    return Model(inputs=encoder.input,
                 outputs=[encoder],
                 name="FINETUNING")
