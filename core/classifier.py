import tensorflow as tf
import json
import os

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from core.astromer import get_ASTROMER
from core.metrics import custom_acc
from core.tboard import save_scalar
from core.losses import custom_bce
from tqdm import tqdm


from core.data import standardize

def get_fc_attention(units, num_classes, weigths):
    ''' FC + ATT'''

    conf_file = os.path.join(weigths, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'])
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    encoder = model.get_layer('encoder')
    encoder.trainable = False

    x = encoder(encoder.input)
    x = tf.keras.layers.Flatten()(x)
    x = standardize(x, axis=1)
    x = Dense(units, name='FCN1')(x)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN2')(x)
    return Model(inputs=encoder.input, outputs=x, name="FCATT")

def get_lstm_no_attention(units, num_classes, maxlen, dropout=0.5):
    ''' LSTM + LSTM + FC'''

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

    bool_mask = tf.logical_not(tf.cast(placeholder['mask_in'], tf.bool))

    x = tf.concat([placeholder['times'], placeholder['input']], 2)

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=placeholder, outputs=x, name="RNNCLF")

def get_lstm_attention(units, num_classes, weigths, dropout=0.5):
    ''' ATT + LSTM + LSTM + FC'''
    conf_file = os.path.join(weigths, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'])
    weights_path = '{}/weights'.format(weigths)
    model.load_weights(weights_path)
    encoder = model.get_layer('encoder')
    encoder.trainable = False

    bool_mask = tf.logical_not(tf.cast(encoder.input['mask_in'], tf.bool))

    x = encoder(encoder.input)
    x = standardize(x, axis=1)

    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_0')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = LSTM(units, return_sequences=True,
             dropout=dropout, name='RNN_1')(x, mask=bool_mask)
    x = LayerNormalization(axis=1)(x)
    x = Dense(num_classes, name='FCN')(x)

    return Model(inputs=encoder.input, outputs=x, name="RNNCLF")

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
        y_pred = model(batch, training=False)
        ce = custom_bce(y_true=batch['label'],
                         y_pred=y_pred)
        acc = custom_acc(batch['label'], y_pred)
    if return_pred:
        return acc, ce, y_pred, batch['label']
    return acc, ce

def train(model,
          train_batches,
          valid_batches,
          patience=20,
          exp_path='./experiments/test',
          epochs=1,
          lr=1e-3,
          verbose=1):
    # Tensorboard
    train_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'train'))
    valid_writter = tf.summary.create_file_writer(
                                    os.path.join(exp_path, 'logs', 'valid'))
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(lr)
    # To save metrics
    train_bce  = tf.keras.metrics.Mean(name='train_bce')
    valid_bce  = tf.keras.metrics.Mean(name='valid_bce')
    train_acc  = tf.keras.metrics.Mean(name='train_acc')
    valid_acc  = tf.keras.metrics.Mean(name='valid_acc')

    # ==============================
    # ======= Training Loop ========
    # ==============================
    best_loss = 999999.
    es_count = 0
    for epoch in range(epochs):
        for train_batch in tqdm(train_batches, desc='train'):
            acc, bce = train_step(model, train_batch, optimizer)
            train_acc.update_state(acc)
            train_bce.update_state(bce)

        for valid_batch in tqdm(valid_batches, desc='validation'):
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
            model.save_weights(os.path.join(exp_path, 'weights'))
        else:
            es_count+=1.
        if es_count == patience:
            print('[INFO] Early Stopping Triggered')
            break

        valid_bce.reset_states()
        train_bce.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()

def predict(model, test_batches):
    predictions = []
    true_labels = []
    for batch in tqdm(test_batches, desc='test'):
        acc, ce, y_pred, y_true = valid_step(model, batch, return_pred=True)
        print(y_pred.shape)
        if len(y_pred.shape)>2:
            predictions.append(y_pred[:, -1, :])
        else:
            predictions.append(y_pred)

        true_labels.append(y_true)

    y_pred = tf.concat(predictions, 0)
    y_true = tf.concat(true_labels, 0)
    pred_labels = tf.argmax(y_pred, 1)

    precision, \
    recall, \
    f1, _ = precision_recall_fscore_support(y_true,
                                            pred_labels,
                                            average='macro')
    acc = accuracy_score(y_true, pred_labels)
    results = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'accuracy':acc}

    return results, y_true, pred_labels

    # os.makedirs(os.path.join(opt.p, 'test'), exist_ok=True)
    # results_file = os.path.join(opt.p, 'test', 'test_results.json')
    # with open(results_file, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
    #
    # h5f = h5py.File(os.path.join(opt.p, 'test', 'predictions.h5'), 'w')
    # h5f.create_dataset('y_pred', data=y_pred.numpy())
    # h5f.create_dataset('y_true', data=y_true.numpy())
