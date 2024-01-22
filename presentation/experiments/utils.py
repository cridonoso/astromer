import tensorflow as tf
import pandas as pd
import pickle
import glob
import toml
import os 

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_mlp_att(inputs, mask, num_cls):
    x = TimeDistributed(Dense(1024, activation='relu'))(inputs)
    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = LayerNormalization(name='layer_norm')(x)
    y_pred = TimeDistributed(Dense(num_cls, name='output_layer'))(x)
    y_pred = tf.reduce_sum(y_pred*mask,1)
    y_pred = tf.math.divide_no_nan(y_pred, tf.reduce_sum(mask, 1))
    return y_pred

def get_linear(inputs, mask, num_cls, invert_mask):
    y_pred = TimeDistributed(Dense(num_cls, name='output_layer'))(inputs)
    y_pred = tf.reduce_sum(y_pred*mask,1)
    y_pred = tf.math.divide_no_nan(y_pred, mask)
    return y_pred

def train_classifier(embedding, inp_placeholder, train_loader, valid_loader, 
    test_loader, num_cls, project_path='', clf_name='emb_dense', debug=False, mask=None):
    os.makedirs(os.path.join(project_path, clf_name), exist_ok=True)
    os.makedirs(os.path.join(project_path, clf_name), exist_ok=True)

    if 'mlp' in clf_name:
        print('[INFO] Training MLP')
        if mask is not None:
            y_pred = get_mlp_att(embedding, mask, num_cls)
        else:
            y_pred = get_mlp_att(embedding, inp_placeholder['att_mask'], num_cls)

    if 'linear' in clf_name:
        print('[INFO] Training Linear')
        if mask is not None:
            y_pred = get_mlp_att(embedding, mask, num_cls)
        else:
            y_pred = get_linear(embedding, inp_placeholder['att_mask'], num_cls)

    classifier = Model(inputs=inp_placeholder, outputs=y_pred, name=clf_name)
    classifier.compile(optimizer=Adam(1e-3),
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    cbks =  [
        ModelCheckpoint(
            filepath=os.path.join(project_path, clf_name, 'weights'),
            save_weights_only=True,
            mode='min',
            monitor='val_loss',
            save_best_only=True),
        EarlyStopping(monitor='val_loss',
            mode='min',
            patience = 20,
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(project_path, clf_name, 'logs'),
            histogram_freq=1,
            write_graph=True)]

    hist = classifier.fit(train_loader,
                         epochs= 2 if debug else 100000,
                         callbacks=cbks,
                         validation_data=valid_loader)

    best_epoch = tf.argmin(hist.history['val_loss'])
    val_loss = hist.history['val_loss'][best_epoch]
    val_acc = hist.history['val_accuracy'][best_epoch]
    y_pred = classifier.predict(test_loader)
    y_true = tf.concat([y for _, y in test_loader], 0)

    with open(os.path.join(project_path, clf_name,'predictions.pkl'), 'wb') as handle:
        pickle.dump({'true':y_true, 'pred':y_pred}, handle)

    pred_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y_true, 1)
    p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                 pred_labels,
                                                 average='macro',
                                                 zero_division=0.)


    test_acc = accuracy_score(true_labels, pred_labels)

    summary_clf = {'clf_val_acc': val_acc,
                   'clf_val_loss': val_loss,
                   'clf_test_precision': p, 
                   'clf_test_recall': r, 
                   'clf_test_f1': f,
                   'clf_test_acc': test_acc}

    with open(os.path.join(project_path, clf_name,'metrics.toml'), 'w') as f:
        toml.dump(summary_clf, f)

    return summary_clf


def get_clf_summary(directory, tag=''):
	clf_folders = glob.glob(os.path.join(directory, 'classification', '*', '*', '*', '*'))

	summary_metrics = []
	for clfdir in clf_folders:
		clf_name = clfdir.split('/')[-1]    
		ds_name  = clfdir.split('/')[-2].split('_')[0]
		spc      = clfdir.split('/')[-2].split('_')[1]
		
		fold_n   = int(clfdir.split('/')[-3].split('_')[-1])
		with open(os.path.join(directory, 'finetuning', ds_name, 'fold_{}'.format(fold_n), 
							  '{}_{}'.format(ds_name, spc), 'config.toml'), 'r') as file:
			config = toml.load(file)

		with open(os.path.join(clfdir, 'metrics.toml'), 'r') as file:
			metrics = toml.load(file)
		metrics['tag'] = tag
		metrics['clf_name'] = clf_name
		metrics['fold'] = fold_n
		metrics['downstream_data'] = ds_name
		metrics['samples_per_class'] = spc
		summary_metrics.append({**config, **metrics})
	df = pd.DataFrame(summary_metrics)
	return df 