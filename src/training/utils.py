import tensorflow as tf
import toml
import time
import os

from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.optimizers import Adam
from src.training.scheduler import CustomSchedule
from tensorflow.keras.optimizers.experimental import AdamW
from src.losses.rmse import custom_rmse

from tqdm import tqdm

def draw_graph(model, dataset, writer, logdir=''):
    '''Decorator that reports store fn graph.'''

    @tf.function
    def fn(x):
        x = model(x)

    tf.summary.trace_on(graph=True, profiler=False)
    fn(dataset)
    with writer.as_default():
        tf.summary.trace_export(
            name='model',
            step=0,
            profiler_outdir=logdir)


def save_scalar(writer, value, step, name=''):
    with writer.as_default():
        tf.summary.scalar(name, value.result(), step=step)
        
@tf.function
def train_step(model, batch, opt):
    x, y = batch
    with tf.GradientTape() as tape:
        x_pred = model(x)

        mse = custom_rmse(y_true=y['target'],
                          y_pred=x_pred,
                          mask=y['mask_out'])


    grads = tape.gradient(mse, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return mse

@tf.function
def valid_step(model, batch, return_pred=False, normed=False):
    x, y = batch
    with tf.GradientTape() as tape:
        x_pred = model(x)
        mse = custom_rmse(y_true=y['target'],
                          y_pred=x_pred,
                          mask=y['mask_out'])

    if return_pred:
        return mse, x_pred, x_true
    return mse

def train(model,
          train_dataset,
          valid_dataset,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          finetuning=False,
          lr=1e-3,
          verbose=1):

    os.makedirs(exp_path, exist_ok=True)

    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'tensorboard', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'tensorboard', 'validation'))

    batch = [x for x, _ in train_dataset.take(1)][0]
    draw_graph(model, batch, train_writter, exp_path)

    # Optimizer
    try:
        d_model = model.get_layer('encoder').num_heads * model.get_layer('encoder').head_dim 
    except:
        d_model = model.get_layer('encoder').d_model

    custom_lr = CustomSchedule(d_model)
    
    optimizer = tf.keras.optimizers.Adam(custom_lr,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    
    # To save metrics
    train_mse  = tf.keras.metrics.Mean(name='train_mse')
    valid_mse  = tf.keras.metrics.Mean(name='valid_mse')

    # Training Loop
    best_loss = 999999.
    es_count = 0
    pbar = tqdm(range(epochs), desc='epoch')
    for epoch in pbar:
        for train_batch in train_dataset:
            x, y  = train_batch
            mse = train_step(model, train_batch, optimizer)
            train_mse.update_state(mse)

        for valid_batch in valid_dataset:
            mse = valid_step(model, valid_batch)
            valid_mse.update_state(mse)

        msg = 'EPOCH {} - ES COUNT: {}/{} train mse: {:.4f} - val mse: {:.4f}'.format(epoch,
                                                                                      es_count,
                                                                                      patience,
                                                                                      train_mse.result(),
                                                                                      valid_mse.result())

        pbar.set_description(msg)

        save_scalar(train_writter, train_mse, epoch, name='mse')
        save_scalar(valid_writter, valid_mse, epoch, name='mse')


        if valid_mse.result() < best_loss:
            best_loss = valid_mse.result()
            es_count = 0.
            model.save_weights(os.path.join(exp_path, 'weights'))
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

    total_mse, inputs, reconstructions = [], [], []
    masks, times = [], []
    for step, batch in tqdm(enumerate(dataset), desc='prediction'):
        mse, x_pred, x_true = valid_step(model,
                                         batch,
                                         return_pred=True,
                                         normed=True)

        total_mse.append(mse)
        times.append(batch['times'])
        inputs.append(x_true)
        reconstructions.append(x_pred)
        masks.append(batch['mask_out'])

    res = {'mse':tf.reduce_mean(total_mse).numpy(),
           'x_pred': tf.concat(reconstructions, 0),
           'x_true': tf.concat(inputs, 0),
           'mask': tf.concat(masks, 0),
           'time': tf.concat(times, 0)}

    return res