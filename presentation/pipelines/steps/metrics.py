import tensorflow as tf
import toml
import os

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.training.scheduler import CustomSchedule

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_ft(astromer, test_loader, params, prefix='test_'):
    metrics = dict()
    if params['arch'] == 'nsp':
        print('[INFO] EVAL NSP FORMAT')
        acc, bce, loss, r2, rmse = astromer.evaluate(test_loader)
        metrics['{}nsp_acc'.format(prefix)] = acc
        metrics['{}nsp_bce'.format(prefix)] = bce
        metrics['{}loss'.format(prefix)]    = loss
        metrics['{}r2'.format(prefix)]      = r2
        metrics['{}rmse'.format(prefix)]    = rmse

    if params['arch'] in ['skip', 'base', 'redux']:
        rmse, r2, loss = astromer.evaluate(test_loader)
        metrics['{}loss'.format(prefix)]    = loss
        metrics['{}r2'.format(prefix)]      = r2
        metrics['{}rmse'.format(prefix)]    = rmse

    return metrics

def evaluate_clf(classifier, test_loader, params, prefix='test_'):
    metrics = dict()


    predictions = classifier.predict(test_loader)

    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    
    pred_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y_true, 1)
    
    p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                 pred_labels,
                                                 average='macro',
                                                 zero_division=0.)
    test_acc = accuracy_score(true_labels, pred_labels)

    metrics['{}acc'.format(prefix)] = test_acc
    metrics['{}precision'.format(prefix)] = p
    metrics['{}recall'.format(prefix)] = r
    metrics['{}f1'.format(prefix)] = f

    return metrics, y_true, y_pred