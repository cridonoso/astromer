import tensorflow as tf
import pandas as pd
import pickle
import glob
import toml
import os 

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def train_classifier(embedding, inp_placeholder, train_loader, valid_loader, test_loader, num_cls, project_path='', clf_name='emb_dense', debug=False):
    os.makedirs(os.path.join(project_path, clf_name), exist_ok=True)
    os.makedirs(os.path.join(project_path, clf_name), exist_ok=True)

    if 'mlp' in clf_name:
        x = Dense(1024, activation='relu')(embedding)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = LayerNormalization(name='layer_norm')(x)
        y_pred = Dense(num_cls, name='output_layer')(x)
        y_pred = tf.reshape(y_pred, [-1, num_cls])
        classifier = Model(inputs=inp_placeholder, outputs=y_pred, name=clf_name)
        classifier.compile(optimizer=Adam(1e-3),
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    
    if 'l2' in clf_name:
        print('L2 Regularizer')
        classifier = tf.keras.Sequential()
        classifier.add(tf.keras.Input(shape=(256)))
        classifier.add(Dense(1024, activation='relu', kernel_regularizer='l2'))
        classifier.add(Dense(512, activation='relu', kernel_regularizer='l2'))
        classifier.add(Dense(256, activation='relu', kernel_regularizer='l2'))
        classifier.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True))
        
    if 'linear':
        x = embedding


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