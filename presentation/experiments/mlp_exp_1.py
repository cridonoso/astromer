import tensorflow as tf
import h5py
import pandas as pd

from tensorflow.keras.layers import (BatchNormalization,
                                     Dense,
                                     LSTM,
                                     LayerNormalization)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import argparse
import os

class generator:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, file):
        with h5py.File(file, 'r') as hf:
            for x, y in zip(hf['embs'], hf['labels']):
                yield x, tf.one_hot(y, self.n_classes)

def standardize(x, y):
    mean_ = tf.reduce_mean(x)
    std_  = tf.math.reduce_std(x)
    x = tf.divide(tf.subtract(x, mean_), std_)
    return x, y

def load_embeddings(path, n_classes, batch_size=16,is_train=False):
    files = [os.path.join(path, x) for x in os.listdir(path)]
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
        generator(n_classes),
        (tf.float32, tf.int32),
        (tf.TensorShape([256]), tf.TensorShape([n_classes])),
        args=(filename,)))
    ds = ds.map(standardize)
    if is_train:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def run(opt):
    df = pd.read_csv(os.path.join(opt.data, 'test_objs.csv'))
    n_classes = len(df['class'].unique())

    train_batches = load_embeddings(os.path.join(opt.data, 'train'),
                                    n_classes, opt.batch_size)
    val_batches = load_embeddings(os.path.join(opt.data, 'val'),
                                    n_classes, opt.batch_size)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(256)))
    model.add(Dense(1024, activation='relu'))
    # model.add(LayerNormalization())
    model.add(Dense(512, activation='relu'))
    # model.add(LayerNormalization())
    model.add(Dense(256, activation='relu'))
    # model.add(LayerNormalization())
    model.add(Dense(n_classes))

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=CategoricalCrossentropy(from_logits=True), metrics='accuracy')
    estop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=20, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(opt.p, 'mlp_att', 'logs'),
        write_graph=False)

    hist = model.fit(train_batches,
                     epochs=opt.epochs,
                     callbacks=[estop, tb],
                     validation_data=val_batches)

    model.save(os.path.join(opt.p, 'mlp_att', 'model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='folder for saving embeddings')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='num of epochs')

    opt = parser.parse_args()
    run(opt)
