import tensorflow as tf
from tqdm import tqdm
import os, sys

from core.output    import RegLayer, ClfLayer, SplitLayer
from core.tboard    import save_scalar, draw_graph
from core.losses    import custom_mse, custom_bce
from core.scheduler import CustomSchedule
from core.metrics   import custom_acc
from core.encoder   import Encoder
from core.decoder   import Decoder


from tensorflow.keras.optimizers.schedules import ExponentialDecay
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

    serie  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(maxlen, 1),
                  batch_size=None,
                  name='mask')
    length = Input(shape=(maxlen,),
                  batch_size=None,
                  dtype=tf.int32,
                  name='length')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times,
                   'length':length}

    x = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=base,
                rate=dropout,
                name='encoder')(placeholder)

    x = RegLayer(name='regression')(x)

    return Model(inputs=placeholder,
                 outputs=x,
                 name="ASTROMER")

@tf.function
def train_step(model, batch, opt):
    with tf.GradientTape() as tape:
        x_pred = model(batch)
        print(x_pred.shape)
        mse = custom_mse(y_true=batch['input'],
                         y_pred=x_pred,
                         mask=batch['mask_out'])


    grads = tape.gradient(mse, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return mse

@tf.function
def valid_step(model, batch, return_pred=False):
    with tf.GradientTape() as tape:
        x_pred = model(batch)

        mse = custom_mse(y_true=batch['input'],
                         y_pred=x_pred,
                         mask=batch['mask_out'])
    if return_pred:
        return 0, 0, 0, mse, x_pred, 0, batch['input'], 0
    return mse

def train(model,
          train_dataset,
          valid_dataset,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          finetuning=False,
          use_random=True,
          num_cls=2,
          lr=1e-3,
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
    optimizer = tf.keras.optimizers.Adam(lr,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    # To save metrics
    train_mse  = tf.keras.metrics.Mean(name='train_mse')
    valid_mse  = tf.keras.metrics.Mean(name='valid_mse')

    # Training Loop
    best_loss = 999999.
    es_count = 0
    for epoch in range(epochs):
        for step, train_batch in tqdm(enumerate(train_dataset), desc='train'):
            mse = train_step(model, train_batch, optimizer)
            train_mse.update_state(mse)

        for valid_batch in tqdm(valid_dataset, desc='validation'):
            mse = valid_step(model, valid_batch)
            valid_mse.update_state(mse)

        save_scalar(train_writter, train_mse, epoch, name='mse')
        save_scalar(valid_writter, valid_mse, epoch, name='mse')


        if verbose == 0:
            print('EPOCH {} - ES COUNT: {}'.format(epoch, es_count))
            print('train mse: {:.2f} - val mse: {:.2f}'.format(train_mse.result(),
                                                               valid_mse.result()))

        if valid_mse.result() < best_loss:
            best_loss = valid_mse.result()
            es_count = 0.
            model.save_weights(exp_path+'/weights')
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break

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

    res = {'loss':tf.reduce_mean(total_loss).numpy(),
           'acc':tf.reduce_mean(total_acc).numpy(),
           'bce':tf.reduce_mean(total_bce).numpy(),
           'mse':tf.reduce_mean(total_mse).numpy(),
           'y_pred':y_pred,
           'x_pred': tf.concat(reconstructions, 0),
           'y_true':tf.concat(true_cls, 0),
           'x_true': tf.concat(true_x, 0)}
    return res
