'''
Experiment to reproduce Donoso et.al., 2022
https://arxiv.org/abs/2205.01677
'''
import tensorflow as tf
import pandas as pd
import tomli
import sys
import os

from src.layers.custom_rnn import NormedLSTMCell, build_zero_init_state 
from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline
from src.training import CustomSchedule

from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.layers     import Dense, Conv1D, Flatten, RNN, LSTM, LayerNormalization
from tensorflow.keras            import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)

from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

def normalize_batch(tensor):
    min_ = tf.expand_dims(tf.reduce_min(tensor, 1), 1)
    max_ = tf.expand_dims(tf.reduce_max(tensor, 1), 1)
    tensor = tf.math.divide_no_nan(tensor - min_, max_ - min_)
    return tensor

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

def get_astromer(config, step='pretraining'):
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # CREATING ASTROMER = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    d_model = config['astromer']['head_dim']*config['astromer']['heads']
    astromer =  get_ASTROMER(num_layers=config['astromer']['layers'],
                             d_model=d_model,
                             num_heads=config['astromer']['heads'],
                             dff=config['astromer']['dff'],
                             base=config['positional']['base'],
                             dropout=config['astromer']['dropout'],
                             maxlen=config['astromer']['window_size'],
                             pe_c=config['positional']['alpha'],
                             no_train=False)
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
                                            cache=config[step]['data']['cache_{}'.format(subset)])
    return data

def train_astromer(astromer, config, step='pretraining', backlog=None, do_train=True):
    cbks = get_callbacks(config, step=step, monitor='val_loss')
    data = load_data(config, step=step)
    
    if do_train:
        _ = astromer.fit(data['train'],
                      epochs=config[step]['epochs'],
                      validation_data=data['val'],
                      callbacks=cbks)
    
    if backlog is not None:
        loss, r2 = astromer.evaluate(data['test'])            
        metrics = {'rmse':loss, 
                   'r_square':r2, 
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
                   'target':config['classification']['data']['target'],
                   'sci_case':'a' if config['classification']['train_astromer'] else 'b', 
                   'spc':config['classification']['data']['spc']
                    }

        metrics = pd.DataFrame(metrics, index=[0])
        backlog = pd.concat([backlog, metrics])

    return clf_model, backlog

def create_classifier(astromer, config, num_cls, name='mlp_att'):
    placeholder = build_input(config['astromer']['window_size'])

    encoder = astromer.get_layer('encoder')
    encoder.trainable = config['classification']['train_astromer']
    z_dim = config['astromer']['head_dim']*config['astromer']['heads']
    n_steps = config['astromer']['window_size']
    conv_shift = 4
    x = encoder(placeholder, training=config['classification']['train_astromer'])
    
    if name == 'lstm':
        print('[INFO] Training an MLP on light curves directly')
        m = tf.cast(1.-placeholder['mask_in'][...,0], tf.bool)
        tim = normalize_batch(placeholder['times'])
        inp = normalize_batch(placeholder['input'])
        x = tf.concat([tim, inp], 2)

        cell_0 = NormedLSTMCell(units=256)
        zero_state = build_zero_init_state(x, 256)
        rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
        x = rnn(x, initial_state=zero_state, mask=m)
        x = tf.nn.dropout(x, .3)
        
    
    if name == 'mlp_att':
        print('[INFO] Training an MLP on time-mean Z')
        mask = 1.-placeholder['mask_in']
        x = x * mask
        x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = LayerNormalization()(x)
    
    if name == 'mlp_att_conv':
        print('[INFO] Training an MLP on convolved Z')
        x = Conv1D(32, 5, activation='relu', input_shape=[n_steps, z_dim])(x)
        x = Flatten()(tf.expand_dims(x, 1))
        x = tf.reshape(x, [-1, (n_steps-conv_shift)*32])

    if name == 'lstm_att':
        print('[INFO] Training an LSTM on Z')
        ssize = 256
        init_states = build_zero_init_state(x, ssize)
        m = tf.cast(1.-placeholder['mask_in'][...,0], dtype=tf.bool)
        x = tf.math.divide_no_nan(x-tf.expand_dims(tf.reduce_mean(x, 1),1),
                                  tf.expand_dims(tf.math.reduce_std(x, 1), 1))
        x = RNN(NormedLSTMCell(units=ssize), 
                return_sequences=False)(x, initial_state=init_states, mask=m) 
        x = tf.nn.dropout(x, .3)
    
    if name == 'mlp_last':
        print('[INFO] Training an MLP on the last position of Z')
        x = tf.slice(x, [0,n_steps-1,0], [-1, 1,-1])
        x = tf.reshape(x, [-1, z_dim])
    if name == 'mlp_first':
        print('[INFO] Training an MLP on the first position of Z')
        x = tf.slice(x, [0,0,0], [-1, 1,-1])
        x = tf.reshape(x, [-1, z_dim])
        
    x = Dense(num_cls)(x)
    return Model(inputs=placeholder, outputs=x, name=name)
    
def pipeline(exp_conf_folder, debug=False, weights_dir=None, load_ft_if_exists=False):
    '''
        Main class to run the pipeline
    '''
    backlog_df = pd.DataFrame(columns=['rmse', 'r_square', 'step', 'time', 'target', 'fold', 'spc'])
    print(backlog_df)
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
        astromer = get_astromer(config, step='pretraining')
        if weights_dir is not None:
            print('[INFO] Loading weigths from: ', weights_dir)
            astromer.load_weights(os.path.join(weights_dir, 'weights'))
            
        if os.path.exists(os.path.join(config['pretraining']['exp_path'], 'checkpoint')): 
            print('[INFO] Checkpoint found! loading weights')
            astromer.load_weights(os.path.join(config['pretraining']['exp_path'], 'weights'))
            if not os.path.exists(os.path.join(config['pretraining']['exp_path'], 'metrics.csv')):
                print('[INFO] Computing metrics ...')
                astromer, backlog_df = train_astromer(astromer, config, 
                                                      step='pretraining', 
                                                      backlog=backlog_df,
                                                      do_train=False)
                backlog_df.to_csv(os.path.join(config['pretraining']['exp_path'], 'metrics.csv'), 
                                  index=False)
            else:
                print('[INFO] Loading metrics...')
                backlog_df = pd.read_csv(os.path.join(config['pretraining']['exp_path'], 'metrics.csv'))
        else:
            print('[INFO] Training from scratch')                    
            astromer, backlog_df = train_astromer(astromer, config, 
                                                  step='pretraining', 
                                                  backlog=backlog_df)
            
            
        # ============================================================================
        # =========== FINETUNING =====================================================
        # ============================================================================
        print('[INFO] Loading weights to finetune')
        if os.path.exists(os.path.join(config['finetuning']['exp_path'], 'checkpoint')) and load_ft_if_exists: 
            print('[INFO] Restoring cktps at: {}'.format(config['finetuning']['exp_path']))
            astromer.load_weights(os.path.join(config['finetuning']['exp_path'], 'weights')).expect_partial()
            metrics_ft = backlog_df[(backlog_df['spc'] == config['finetuning']['data']['spc']) & \
                                    (backlog_df['target'] == config['finetuning']['data']['target']) & \
                                    (backlog_df['fold'] == config['finetuning']['data']['fold']) ]
            print(metrics_ft)
        else:
            astromer.load_weights(os.path.join(config['finetuning']['weights'], 'weights')).expect_partial()
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
                    cache=config['classification']['data']['cache_{}'.format(subset)])

        # 'mlp_att', 'mlp_att_conv', 'lstm'
        for name in ['mlp_last', 'lstm_att', 'mlp_first', 'lstm', 'mlp_att']:
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
    pipeline(exp_conf_folder, debug=False, weights_dir=w, load_ft_if_exists=True)
