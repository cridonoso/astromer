'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import tensorflow as tf
import pandas as pd
import tomli
import sys
import os

from src.data import pretraining_pipeline
from src.training import CustomSchedule
from src.models import get_ASTROMER_II, build_input

from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.layers     import Dense, Conv1D, Flatten
from tensorflow.keras            import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)

from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support


def get_callbacks(config, step='pretraining', monitor='val_loss', extra=''):
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config[step]['exp_path'], extra, 'weights'),
            save_weights_only=True,
            monitor=monitor,
            save_best_only=True),
        EarlyStopping(monitor=monitor,
            patience = config[step]['patience'],
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(config[step]['exp_path'], extra, 'logs'),
            histogram_freq=1,
            write_graph=True)]
    return callbacks

def build_astromer(config, step='pretraining'):
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # CREATING ASTROMER = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    d_model = config['astromer']['head_dim']*config['astromer']['heads']
    astromer =  get_ASTROMER_II(num_layers=config['astromer']['layers'],
                                d_model=d_model,
                                num_heads=config['astromer']['heads'],
                                dff=config['astromer']['dff'],
                                base=config['positional']['base'],
                                dropout=config['astromer']['dropout'],
                                maxlen=config['astromer']['window_size'],
                                pe_c=config['positional']['alpha'])
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # OPTIMIZER = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    if not config[step]['scheduler']:
        print('[INFO] Using learning rate: {}'.format(config[step]['lr']))
        lr = config[step]['lr']
    else:
        print('[INFO] Using custom scheduler')
        model_dim = config['astromer']['head_dim']*config['astromer']['heads']
        lr = CustomSchedule(model_dim)

    optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    astromer.compile(optimizer=optimizer)
    return astromer
        
def load_data(config, step='pretraining'):
    print('[INFO] Loading data from {}'.format(config[step]['data']['path']))
    print('[INFO] Batch size: ', config[step]['data']['batch_size'])
    data = dict()
    for subset in ['train', 'val', 'test']:
        repeat = config[step]['data']['repeat'] if subset == 'train' else None
        data[subset] = pretraining_pipeline(os.path.join(config[step]['data']['path'], subset),
                                            config[step]['data']['batch_size'],
                                            config['astromer']['window_size'],
                                            config['masking']['mask_frac'],
                                            config['masking']['rnd_frac'],
                                            config['masking']['same_frac'],
                                            config[step]['data']['sampling'],
                                            config[step]['data']['shuffle_{}'.format(subset)],
                                            repeat=repeat,
                                            num_cls=None,
                                            normalize=config[step]['data']['normalize'],
                                            cache=config[step]['data']['cache_{}'.format(subset)],
                                            nsp_prob=config['nsp']['probability'],
                                            nsp_frac=config['nsp']['fraction'],
                                            moving_window=False)
    return data

def train_astromer(astromer, config, step='pretraining', backlog=None):
    cbks = get_callbacks(config, step=step, monitor='val_loss')
    data = load_data(config, step=step)

    _ = astromer.fit(data['train'],
                  epochs=config[step]['epochs'],
                  validation_data=data['val'],
                  callbacks=cbks)
    
    if backlog is not None:
        acc, bce, loss, r2, rmse = astromer.evaluate(data['val'])            
        metrics = {'loss':loss,
                   'rmse':rmse, 
                   'r_square':r2,
                   'acc':acc, 
                   'bce':bce,
                   'step': step, 
                   'time': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                   'target':config[step]['data']['target'], 
                   'fold':config[step]['data']['fold'], 
                   'spc':config[step]['data']['spc']}

        metrics = pd.DataFrame(metrics, index=[0])
        backlog = pd.concat([backlog, metrics])

    return astromer, backlog


def classify(clf_model, data, config, backlog=None, model_name='mlp'):

    # Compile and train
    optimizer = Adam(learning_rate=config['classification']['lr'])
    exp_path_clf = config['classification']['exp_path']
    os.makedirs(exp_path_clf, exist_ok=True)

    clf_model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics='accuracy')

    cbks = get_callbacks(config, step='classification',
                         monitor='val_loss', extra=model_name)

    history = clf_model.fit(data['train'],
                            epochs=config['classification']['epochs'],
                            callbacks=cbks,
                            validation_data=data['val'])

    clf_model.save_weights(os.path.join(exp_path_clf, clf_model.name, 'model'))

    if backlog is not None:
        y_pred = clf_model.predict(data['test'])
        y_true = tf.concat([y for _, y in data['test']], 0)

        pred_labels = tf.argmax(y_pred, 1)
        true_labels = tf.argmax(y_true, 1)

        p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                     pred_labels,
                                                     average='macro',
                                                     zero_division=0.)

        metrics = {'test_precision':p, 
                   'test_recall':r, 
                   'test_f1': f,
                   'val_acc': tf.reduce_max(history.history['val_accuracy']).numpy(),
                   'val_loss': tf.reduce_min(history.history['val_loss']).numpy(),
                   'model':clf_model.name,
                   'time': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                   'fold':config['classification']['data']['fold'], 
                   'sci_case':'a' if config['classification']['train_astromer'] else 'b', 
                   'spc':config['classification']['data']['spc']
                    }

        metrics = pd.DataFrame(metrics, index=[0])
        backlog = pd.concat([backlog, metrics])

    return clf_model, backlog

def create_classifier(astromer, config, num_cls, name='mlp', mode=0):
    placeholder = build_input(config['astromer']['window_size']+3)

    encoder = astromer.get_layer('encoder')
    encoder.trainable = config['classification']['train_astromer']
    z_dim = config['astromer']['head_dim']*config['astromer']['heads']
    n_steps = config['astromer']['window_size'] + 3
    conv_shift = 4
    
    x = encoder(placeholder, training=config['classification']['train_astromer'])
    if name == 'mlp_cls':
        x = tf.slice(x, [0,0,0], [-1,1,-1], name='cls_tkn')
        x = tf.reshape(x, [-1, z_dim])

    if name == 'mlp_att':
        x = tf.slice(x, [0,1,0], [-1,-1,-1], name='att_tkn')
        conv_shift+=1

    if 'att' in name:
        x = Conv1D(32, 5, activation='relu', input_shape=[n_steps, z_dim])(x)
        x = Flatten()(tf.expand_dims(x, 1))
        x = tf.reshape(x, [-1, (n_steps-conv_shift)*32])
    
    x = Dense(num_cls)(x)
    return Model(inputs=placeholder, outputs=x, name=name)

# =================================================================
# =================================================================
# =================================================================
# PIPELINE BEINGS =================================================
# =================================================================
# =================================================================
# =================================================================
def pipeline(exp_conf_folder, debug=False, weights_dir=None):
    '''
        Main class to run the pipeline
    '''
    # to save partial results
    backlog_df = pd.DataFrame(columns=['loss','rmse', 'r_square', 'acc', 'bce',
                                       'step', 'time', 'target', 'fold', 'spc'])

    for config_file in os.listdir(exp_conf_folder):
        if not config_file.endswith('toml'): continue
        # Load config file
        with open(os.path.join(exp_conf_folder, config_file), mode="rb") as fp:
            config = tomli.load(fp)

        if debug:
            config['pretraining']['data']['batch_size'] = 32
            config['pretraining']['epochs'] = 1
            config['finetuning']['data']['batch_size'] = 32
            config['finetuning']['epochs'] = 1
            config['classification']['data']['batch_size'] = 32
            config['classification']['epochs'] = 1

        # ============================================================================
        # =========== PRETRAINING ====================================================
        # ============================================================================    
        astromer = build_astromer(config, step='pretraining')
        if weights_dir is not None:
            print('[INFO] Loading weigths from: ', weights_dir)
            astromer.load_weights(os.path.join(weights_dir, 'weights'))
            

        if not os.path.exists(os.path.join(config['pretraining']['exp_path'], 'metrics.csv')):    
            print('[INFO] Training from scratch')                    
            astromer, backlog_df = train_astromer(astromer, config, 
                                                  step='pretraining', 
                                                  backlog=backlog_df)
        else:
            backlog_df = pd.read_csv(os.path.join(config['pretraining']['exp_path'], 'metrics.csv'))
            

        # ============================================================================
        # =========== FINETUNING =====================================================
        # ============================================================================
        print('[INFO] Loading weights to finetune')
        astromer.load_weights(os.path.join(config['finetuning']['weights'], 'weights'))
        astromer, backlog_df = train_astromer(astromer, config, 
                                              step='finetuning', 
                                              backlog=backlog_df)
        backlog_df.to_csv(os.path.join(config['pretraining']['exp_path'], 'metrics.csv'), 
                          index=False)
        # ============================================================================
        # =========== CLASSIFICATION==================================================
        # ============================================================================
        exp_path_root = '/'.join(config['classification']['exp_path'].split('/')[:-2])
        
        if os.path.exists(os.path.join(exp_path_root, 'metrics.csv')):
            backlog_df = pd.read_csv(os.path.join(exp_path_root, 'metrics.csv'))
        else:
            backlog_df = pd.DataFrame(columns=['test_precision', 'test_recall', 'test_f1', 'val_acc',
                                               'val_loss', 'model','time', 'fold', 'sci_case', 'spc'])

        num_cls = pd.read_csv(
                os.path.join(config['classification']['data']['path'],
                            'objects.csv')).shape[0]

        data = dict()
        for subset in ['train', 'val', 'test']:
            repeat = config['classification']['data']['repeat'] if subset == 'train' else None
            data[subset] = pretraining_pipeline(
                    os.path.join(config['classification']['data']['path'], subset),
                    config['classification']['data']['batch_size'],
                    config['astromer']['window_size'],
                    0.,
                    0.,
                    0.,
                    False,
                    config['classification']['data']['shuffle_{}'.format(subset)],
                    repeat=repeat,
                    num_cls=num_cls,
                    normalize=config['classification']['data']['normalize'],
                    cache=config['classification']['data']['cache_{}'.format(subset)],
                    nsp_prob=0.,
                    nsp_frac=.5)

        for name in ['mlp_att', 'mlp_cls_att', 'mlp_cls']:
            print('[INFO] Training {} classifier'.format(name))
            clf_model = create_classifier(astromer, config, num_cls=num_cls, name=name)

            clf_model, backlog_df = classify(clf_model, data, config, 
                                             backlog=backlog_df, 
                                             model_name=clf_model.name)
                    
            backlog_df.to_csv(os.path.join(exp_path_root, 'metrics.csv'), 
                           index=False)


if __name__ == '__main__':
    exp_conf_folder = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    try:
        w = sys.argv[3] # preloading weights
    except:
        w = None

    pipeline(exp_conf_folder, debug=True, weights_dir=w)
