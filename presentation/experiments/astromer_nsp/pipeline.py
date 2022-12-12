'''
NSP
'''
import tensorflow as tf
import pandas as pd
import tomli
import os, sys

from presentation.experiments.astromer_nsp.classifiers import get_classifier_by_name
from src.data import pretraining_pipeline
from src.pipeline import base
from src.models import get_ASTROMER_nsp
from time import gmtime, strftime, time

from tensorflow.keras.callbacks  import EarlyStopping, TensorBoard
from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_recall_fscore_support

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

def train(config_file, step='pretraining', testing=False):
    '''
    pretraining/finetuning pipeline
    '''
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    base.create_target_directory(config_file=config_file,
                            path=config[step]['exp_path'])

    # Creating ASTROMER
    d_model = config['astromer']['head_dim']*config['astromer']['heads']
    astromer =  get_ASTROMER_nsp(num_layers=config['astromer']['layers'],
                                 d_model=d_model,
                                 num_heads=config['astromer']['heads'],
                                 dff=config['astromer']['dff'],
                                 base=config['positional']['base'],
                                 dropout=config['astromer']['dropout'],
                                 maxlen=config['astromer']['window_size'])
    astromer = base.compile_astromer(config, astromer, step=step)

    # Get callbacks
    cbks = base.get_callbacks(config, step=step, monitor='val_loss')

    # Loading data
    data = base.load_pt_data(config, subsets=['train', 'val', 'test'], step=step,
                             nsp_prob=.5, nsp_frac=.5)

    # Train ASTROMER
    _ = astromer.fit(data['train'],
                  epochs=config[step]['epochs'],
                  validation_data=data['val'],
                  callbacks=cbks)

    # Getting metrics
    try:
        loss, r2 = astromer.evaluate(data['test'])
        metrics = {'rmse':loss, 'r_square':r2}
    except:
        acc, bce, loss, r2, rmse = astromer.evaluate(data['test'])
        metrics = {'loss':loss, 'rmse':rmse, 'r_square':r2, 'bce':bce, 'acc':acc}

    base.save_metrics(metrics,
                 path=os.path.join(config[step]['exp_path'],
                                   'metrics.csv'))

def classify(config_file):
    step = 'classification'
    start = time()
    with open(config_file, mode="rb") as fp:
        config = tomli.load(fp)

    base.create_target_directory(config_file=config_file,
                            path=config['classification']['exp_path'])

    # Load data for classification
    num_cls = pd.read_csv(
            os.path.join(config[step]['data']['path'],
                        'objects.csv')).shape[0]
    train_data = pretraining_pipeline(os.path.join(config[step]['data']['path'], 'train'),
                                      config[step]['data']['batch_size'],
                                      config['astromer']['window_size'],
                                      0.,
                                      0.,
                                      0.,
                                      config[step]['data']['sampling'],
                                      config[step]['data']['shuffle_train'],
                                      repeat=1,
                                      num_cls=num_cls,
                                      normalize=config[step]['data']['normalize'],
                                      cache=config[step]['data']['cache_train'],
                                      nsp_prob=0.,
                                      nsp_frac=.5,
                                      moving_window=False)

    valid_data = pretraining_pipeline(os.path.join(config[step]['data']['path'], 'train'),
                                      config[step]['data']['batch_size'],
                                      config['astromer']['window_size'],
                                      0.,
                                      0.,
                                      0.,
                                      config[step]['data']['sampling'],
                                      config[step]['data']['shuffle_train'],
                                      repeat=1,
                                      num_cls=num_cls,
                                      normalize=config[step]['data']['normalize'],
                                      cache=config[step]['data']['cache_train'],
                                      nsp_prob=0.,
                                      nsp_frac=.5,
                                      moving_window=False)
    
    test_data = pretraining_pipeline(os.path.join(config[step]['data']['path'], 'test'),
                                  config[step]['data']['batch_size'],
                                  config['astromer']['window_size'],
                                  0.,
                                  0.,
                                  0.,
                                  config[step]['data']['sampling'],
                                  config[step]['data']['shuffle_test'],
                                  repeat=1,
                                  num_cls=num_cls,
                                  normalize=config[step]['data']['normalize'],
                                  cache=config[step]['data']['cache_test'],
                                  nsp_prob=0.,
                                  nsp_frac=.5,
                                  moving_window=False)


    for clf_name in ['mlp_cls', 'lstm', 'mlp_att_cls', 'mlp_att', 'lstm_att']:
        print('[INFO] Training {}'.format(clf_name))
        # Load pre-trained model
        d_model = config['astromer']['head_dim']*config['astromer']['heads']
        astromer =  get_ASTROMER_nsp(num_layers=config['astromer']['layers'],
                                 d_model=d_model,
                                 num_heads=config['astromer']['heads'],
                                 dff=config['astromer']['dff'],
                                 base=config['positional']['base'],
                                 dropout=config['astromer']['dropout'],
                                 maxlen=config['astromer']['window_size'])

        astromer = base.compile_astromer(config, astromer, step='classification')

        # Create classifier
        clf_model = get_classifier_by_name(clf_name,
                    config,
                    astromer=astromer,
                    train_astromer=config['classification']['train_astromer'])

        # Compile and train
        optimizer = Adam(learning_rate=config['classification']['lr'])
        exp_path_clf = config['classification']['exp_path']
        os.makedirs(exp_path_clf, exist_ok=True)

        clf_model.compile(optimizer=optimizer,
                          loss=CategoricalCrossentropy(from_logits=True),
                          metrics='accuracy')

        cbks = base.get_callbacks(config, step='classification',
                             monitor='val_loss', extra=clf_name)

        history = clf_model.fit(train_data,
                                epochs=config['classification']['epochs'],
                                callbacks=cbks,
                                validation_data=valid_data)
        clf_model.save(os.path.join(exp_path_clf, clf_name, 'model'))

        # Evaluate
        y_pred = clf_model.predict(test_data)
        y_true = tf.concat([y for _, y in test_data], 0)

        pred_labels = tf.argmax(y_pred, 1)
        true_labels = tf.argmax(y_true, 1)

        p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                     pred_labels,
                                                     average='macro')
        metrics = {'precision':p, 'recall':r, 'f1': f,
                   'val_acc': tf.reduce_max(history.history['val_accuracy']).numpy(),
                   'val_loss': tf.reduce_min(history.history['val_loss']).numpy(),
                   'model':clf_name}
        # # Save metrics
        base.save_metrics(metrics, path=os.path.join(exp_path_clf, 'metrics.csv'))
        


if __name__ == '__main__':

    directory = sys.argv[1]
    mode = sys.argv[3] # pretraining - finetuning - classification
    print('[INFO] Mode: {}'.format(mode))
    if directory.endswith('.toml'):
        print('[INFO] Single file recieved')
        conf_files = [directory]
    else:
        conf_files = [os.path.join(directory, d) for d in os.listdir(directory)]

    for config_file in conf_files:
        if mode == 'classification':
            classify(config_file)
        if mode == 'pretraining' or mode == 'finetuning':
            train(config_file, step=mode)
        if mode == 'testing':
            test_classification(config_file)
            
