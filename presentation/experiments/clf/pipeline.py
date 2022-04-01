import tensorflow as tf
import pandas as pd
import subprocess
import argparse
import json
import os, sys

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from core.data import load_dataset, inference_pipeline

from presentation.experiments.clf.classifiers import build_lstm, \
                                                     build_lstm_att, \
                                                     build_mlp_att
from core.astromer import ASTROMER


def finetuning(datapath, astroweights, batch_size, project_path, gpu):
    command1 = 'python -m presentation.scripts.train \
               --data {} \
               --w {} \
               --batch-size {} \
               --p {} \
               --gpu {}'.format(datapath,
                                astroweights,
                                batch_size,
                                project_path,
                                gpu)
    subprocess.call(command1, shell=True)

def classify(datapath, astroweights, batch_size, project_path, model_arch='lstm'):
    max_obs = 200
    n_classes = pd.read_csv(os.path.join(datapath, 'objects.csv')).shape[0]
    dataset = load_dataset(os.path.join(datapath, 'train'),repeat=1)
    train_batches = inference_pipeline(dataset,
                                 batch_size=batch_size,
                                 max_obs=max_obs,
                                 n_classes=n_classes,
                                 shuffle=True,
                                 mode='clf')

    dataset = load_dataset(os.path.join(datapath, 'val'),repeat=1)
    val_batches = inference_pipeline(dataset,
                                 batch_size=batch_size,
                                 max_obs=max_obs,
                                 n_classes=n_classes,
                                 shuffle=True,
                                 mode='clf')

    if model_arch == 'mlp_att':
        astromer = ASTROMER()
        astromer.build(batch_size=batch_size, max_obs=200, inp_dim=1)
        astromer.load_weights(astroweights)
        model = build_mlp_att(astromer, max_obs, n_classes=n_classes,
                              train_astromer=False)

    if model_arch == 'lstm_att':
        astromer = ASTROMER()
        astromer.build(batch_size=batch_size, max_obs=200, inp_dim=1)
        astromer.load_weights(astroweights)

        model = build_lstm_att(astromer, max_obs, n_classes=n_classes,
                               train_astromer=False)

    if model_arch == 'lstm':
        model = build_lstm(max_obs, n_classes=n_classes)

    optimizer = Adam(learning_rate=1e-3)

    os.makedirs(project_path, exist_ok=True)
    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    estop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=40,
                          verbose=0,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)
    tb = TensorBoard(log_dir=os.path.join(project_path, 'logs'),
                     write_graph=False,
                     write_images=False,
                     update_freq='epoch',
                     profile_batch=0,
                     embeddings_freq=0,
                     embeddings_metadata=None)

    _ = model.fit(train_batches,
                  epochs=10000,
                  callbacks=[estop, tb],
                  validation_data=val_batches)

    model.save(os.path.join(project_path, 'weights'))

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    finetuning(opt.data, opt.w, 2500, opt.p, opt.gpu)

    for model_arch in ['lstm', 'lstm_att', 'mlp_att']:
        project_path = os.path.join(opt.p, model_arch)
        classify(opt.data, opt.p, 512, project_path, model_arch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--data', default='./data/records/ogle/fold_0/ogle_20', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--w', default='./weights/macho', type=str,
                        help='ASTROMER pretrained weights')
    parser.add_argument('--p', default='./experiments/clf/ogle/fold_0/ogle_20', type=str,
                        help='ASTROMER pretrained weights')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')
    opt = parser.parse_args()
    run(opt)
