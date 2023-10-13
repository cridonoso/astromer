import keras
import pandas as pd
import os, sys
import pickle
import tensorflow as tf
from src.data import pretraining_pipeline, load_data
from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import LayerNormalization, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from src.models.astromer_1 import get_ASTROMER, train_step, test_step, build_input
import argparse
import toml
from src.training.utils import train
from src.data import pretraining_pipeline, load_data

from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)


from sklearn.metrics import precision_recall_fscore_support, accuracy_score




def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] =  opt.gpu
    DOWNSTREAM_DATA = './records/{}/fold_{}/{}_{}'.format(opt.subdataset, opt.fold, opt.subdataset, opt.spc)
    
    if opt.train_astromer: tag = 'trainable'
    else: tag = 'frozen'
    CLFWEIGHTS = os.path.join(opt.pt_folder,
                                  'classification',
                                   tag, 
                                   opt.subdataset, 
                                  'fold_'+str(opt.fold), 
                                    opt.subdataset+'_'+str(opt.spc))

    FTWEIGTHS = os.path.join(opt.pt_folder,
                                    'finetuning',                                     
                                    opt.subdataset,
                                    'fold_'+str(   opt.fold), 
                                    '{}_{}'.format(   opt.subdataset,    opt.spc))

    num_cls = pd.read_csv(os.path.join(DOWNSTREAM_DATA, 'objects.csv')).shape[0]

    print('DOWNSTREAM_DATA', DOWNSTREAM_DATA)
    print('CLFWEIGHTS', CLFWEIGHTS)
    print('FTWEIGTHS',FTWEIGTHS)


    with open(os.path.join(opt.pt_folder, 'model_config.toml'), 'r') as f:
            model_config = toml.load(f)
        
    astromer = get_ASTROMER(num_layers=model_config['num_layers'], 
                                num_heads=model_config['num_heads'], 
                                head_dim=model_config['head_dim'],
                                mixer_size=model_config['mixer'],
                                dropout=model_config['dropout'],
                                pe_base=model_config['pe_base'], 
                                pe_dim=model_config['pe_dim'],
                                pe_c=model_config['pe_exp'],
                                window_size=model_config['window_size'],
                                encoder_mode=model_config['encoder_mode'],
                                average_layers=model_config['avg_layers']
                                )
    
    astromer.load_weights(os.path.join(FTWEIGTHS, 'weights', 'weights'))

    inp_placeholder = build_input(200)
    encoder = astromer.get_layer('encoder')
    embedding = encoder(inp_placeholder)
    if opt.train_astromer is False:
        encoder.trainable = False
        embedding = encoder(inp_placeholder,training=False)
        
    embedding = embedding*(1.-inp_placeholder['att_mask'])
    embedding = tf.math.divide_no_nan(tf.reduce_sum(embedding, axis=1),
                        tf.reduce_sum(1.-inp_placeholder['att_mask'], axis=1))
    
    if 'mlp' in opt.clf_name:
        x = Dense(1024, activation='relu')(embedding)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
    else:
        x = embedding

    x      = LayerNormalization(name='layer_norm')(x)
    y_pred = Dense(num_cls, name='output_layer')(x)

    classifier = Model(inputs=inp_placeholder, outputs=y_pred)
    
    # LOADING DATA
    BATCH_SIZE = 512
    print('BATCH_SIZE', BATCH_SIZE)

    train_loader = load_data(dataset='{}/train'.format(DOWNSTREAM_DATA),
                                        batch_size=BATCH_SIZE,
                                        window_size=200,
                                        probed=1. , 
                                        random_same=0 ,
                                        sampling=True,
                                        off_nsp=True, 
                                        repeat=4, num_cls=num_cls)
    valid_loader = load_data(dataset='{}/val'.format(DOWNSTREAM_DATA),
                                        batch_size=BATCH_SIZE,
                                        window_size=200,
                                        off_nsp=True, 
                                        probed=1.,
                                        random_same=0 ,
                                        sampling=True,
                                        repeat=1,
                                        num_cls=num_cls)
    test_loader = load_data(dataset='{}/test'.format(DOWNSTREAM_DATA), 
                                batch_size=BATCH_SIZE, 
                                probed=1,  
                                random_same=0 ,
                                window_size=200, 
                                off_nsp=True, 
                                repeat=1, 
                                sampling=True, num_cls=num_cls)

    classifier.compile(optimizer=Adam(1e-3),
                            loss=CategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
    cbks =  [
            ModelCheckpoint(
            filepath=os.path.join(CLFWEIGHTS, opt.clf_name, 'weights'),
            save_weights_only=True,
                    mode='min',
                    monitor='val_loss',
                    save_best_only=True),
                EarlyStopping(monitor='val_loss',
                    mode='min',
                    patience = 20,
                    restore_best_weights=True),
                TensorBoard(
                    log_dir = os.path.join(CLFWEIGHTS, opt.clf_name, 'logs'),
                    histogram_freq=1,
                    write_graph=True)]

    hist = classifier.fit(train_loader,
                                epochs=  100000,
                                callbacks=cbks,
                                validation_data=valid_loader)

    best_epoch = tf.argmin(hist.history['val_loss'])
    val_loss = hist.history['val_loss'][best_epoch]
    val_acc = hist.history['val_accuracy'][best_epoch]
    y_pred = classifier.predict(test_loader)
    y_true = tf.concat([y for _, y in test_loader], 0)

    with open(os.path.join(CLFWEIGHTS, opt.clf_name,'predictions.pkl'), 'wb') as handle:
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

    with open(os.path.join(CLFWEIGHTS, opt.clf_name,'metrics.toml'), 'w') as f:
        toml.dump(summary_clf, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='0', type=str, help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--subdataset', default='alcock', type=str, help='Data folder where tf.record files are located')
	parser.add_argument('--pt-folder', default='./results/pretraining*', type=str, help='pretrained model folder')
	parser.add_argument('--fold', default=0, type=int, help='Fold to use')
	parser.add_argument('--spc', default=20, type=int, help='Samples per class')


	parser.add_argument('--train-astromer', action='store_true', help='If train astromer when classifying')
	parser.add_argument('--clf-name', default='att_mlp', type=str, help='classifier name')

	opt = parser.parse_args()        
	run(opt)