import tensorflow as tf 
import wandb
import sys
import json
import pandas as pd
import os

from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline
from src.training import CustomSchedule

from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from src.layers.custom_rnn import NormedLSTMCell, build_zero_init_state 

from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.layers     import Dense, Conv1D, Flatten, RNN, LSTM, LayerNormalization
from tensorflow.keras            import Input, Model
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)
from sklearn.metrics import precision_recall_fscore_support

# inciwpm3
MASTER_PROJECT_NAME = 'downstream'
WEIGHTS_FOLDER = './presentation/scripts/hp_results'
os.makedirs(os.path.join(WEIGHTS_FOLDER, MASTER_PROJECT_NAME), exist_ok=True)

# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
    'name': 'ASTROMER_I',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'epoch/val_accuracy'},
    'parameters': {
        'paths':{'values':['masking/0.20', 'masking/0.40', 'masking/0.50', 'masking/0.60', 'masking/0.80', 'masking/1.00',
                 'winsize/100', 'winsize/200', 'winsize/500', 'winsize/800']},
        'n_layers': {'values':[1]},
        'fold':{'values':[0, 1, 2]},
        'dataset_to_ft':{'values':['atlas', 'alcock']},
        'clf_name':{'values':['mlp_att', 'mlp_att_conv', 'mlp_last', 'mlp_first']},
        'n_heads': {'value':4},
        'head_dim': {'value':64},
        'dff': {'value':64},
        'dropout_rate': {'value': 0.3955},
        'learning_rate':{'value':1e-5},
        'rand': {'value':0.2}
    }
}

def get_batch_size(model, bytes_per_param=4, window_size=None):
    params = model.count_params()    
    if window_size > 200:
        bs = int(300*595841/params)
    else:
        bs = int(3000*595841/params)
    return min(bs, 3000)

def load_clf_data(path, batch_size, window_size, num_cls, step='classification', debug=False):
    data = dict()
    for subset in ['train', 'val', 'test']:
        data[subset] = pretraining_pipeline(
                os.path.join(path, subset),
                batch_size,
                window_size,
                0.,
                0.,
                0.,
                False,
                True,
                num_cls=num_cls,
                normalize='zero-mean',
                cache=True)
        if debug:
            data[subset] = data[subset].take(1)

    return data

def get_callbacks(path, step='pretraining', monitor='val_loss', extra=''):
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(path, extra, 'weights'),
            save_weights_only=True,
            monitor=monitor,
            save_best_only=True),
        EarlyStopping(monitor=monitor,
            patience = 20,
            restore_best_weights=True),
        TensorBoard(
            log_dir = os.path.join(path, extra, 'logs'),
            histogram_freq=1,
            write_graph=True),
        WandbMetricsLogger(log_freq='epoch')]
    return callbacks


def classify(clf_model, data, exp_path_clf, lr=1e-5, model_name='mlp'):

    # Compile and train
    optimizer = Adam(learning_rate=lr)
    os.makedirs(exp_path_clf, exist_ok=True)

    clf_model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics='accuracy')

    cbks = get_callbacks(os.path.join(exp_path_clf, clf_model.name), step='classification',
                         monitor='val_loss', extra=model_name)
    
    history = clf_model.fit(data['train'],
                            epochs=100000,
                            callbacks=cbks,
                            validation_data=data['val'])

    clf_model.save_weights(os.path.join(exp_path_clf, clf_model.name, 'model'))

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
               'test_f1': f}

    return clf_model, metrics

def create_classifier(astromer, window_size, heads=4, head_dim=64, num_cls=None, train_astromer=False, name='mlp_att'):
    placeholder = build_input(window_size)

    encoder = astromer.get_layer('encoder')
    encoder.trainable = train_astromer
    
    z_dim = head_dim*heads
    n_steps = window_size
    conv_shift = 4
    x = encoder(placeholder, training=train_astromer)
    
    if name == 'lstm':
        print('[INFO] Training an MLP on light curves directly')
        m = tf.cast(1.-placeholder['mask_in'][...,0], tf.bool)
        tim = normalize_batch(placeholder['times'])
        inp = normalize_batch(placeholder['input'])
        x = tf.concat([tim, inp], 2)

        cell_0 = NormedLSTMCell(units=256)
        zero_state = build_zero_init_state(x, 256)
        rnn = tf.keras.layers.RNN(cell_0, return_sequences=False)
        drop_layer = tf.keras.layers.Dropout(.3)
        
        x = rnn(x, initial_state=zero_state, mask=m)
        x = drop_layer(x)
        
    
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
        x = tf.keras.layers.Dropout(.3)(x)
    
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

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        api = wandb.Api()
        runs = api.runs(MASTER_PROJECT_NAME)
        ft_done = False
        for run in runs:
            if run.config['paths'] == config.paths and \
               run.config['fold']== config.fold and \
               run.config['dataset_to_ft'] == config.dataset_to_ft and \
               run.state == 'finished':
                ft_done=True
                break
        
        project_name = str(config.paths).split('/')[0]
        hpcurr = str(config.paths).split('/')[1]
        
        if project_name == 'winsize':
            PTWEIGTHS = os.path.join(WEIGHTS_FOLDER, project_name, str(hpcurr))
            window_size = int(hpcurr)
            probed = 0.50
        else: 
            PTWEIGTHS = os.path.join(WEIGHTS_FOLDER, project_name, str(hpcurr))
            window_size = 200 
            probed = hpcurr
            
        
        wandb.log({"window_size": window_size, "probed": probed})
        # =====================================================================================
        # === data =========================
        # =====================================================================================
        curr_dataset = os.path.join('./data/records', config.dataset_to_ft,'fold_'+str(config.fold), config.dataset_to_ft+'_20')

        # =====================================================================================
        # ===== MODEL =========================================================================
        # =====================================================================================
        d_model      = config.head_dim*config.n_heads
        astromer 	 =  get_ASTROMER(num_layers=config.n_layers,
                                     d_model=d_model,
                                     num_heads=config.n_heads,
                                     dff=config.dff,
                                     base=1000,
                                     dropout=config.dropout_rate,
                                     maxlen=window_size,
                                     pe_c=2.,
                                     no_train=False)

        batch_size = get_batch_size(astromer, window_size=window_size)

        # =====================================================================================
        # ===== DATA ==========================================================================
        # =====================================================================================

        # -------------------------------------------------------------------------------------
        trainloader = pretraining_pipeline(os.path.join(curr_dataset, 'train'), 
                                           batch_size, 
                                           window_size, 
                                           float(probed), config.rand, config.rand,
                                           sampling=False, 
                                           shuffle=True, 
                                           repeat=4, 
                                           num_cls=None,
                                           normalize="zero-mean", 
                                           cache=True)
        validloader = pretraining_pipeline(os.path.join(curr_dataset, 'val'), 
                                           batch_size, 
                                           window_size, 
                                           float(probed), config.rand, config.rand,
                                           sampling=False, 
                                           shuffle=False, 
                                           repeat=1, 
                                           num_cls=None,
                                           normalize="zero-mean", 
                                           cache=True)

        # =====================================================================================
        # ===== TRAINING ======================================================================
        # =====================================================================================
        lr = 1e-5 #CustomSchedule(d_model)
        optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        astromer.compile(optimizer=optimizer)
        astromer.load_weights(os.path.join(PTWEIGTHS, 'weights')).expect_partial()
        N_EPOCHS = 10000
        
        # CHECK IF THERE IS ALREADY FINETUNED WEIGHTS
        SAVEPATH = os.path.join(PTWEIGTHS, 'finetuning', config.dataset_to_ft,'fold_'+str(config.fold), config.dataset_to_ft+'_20')
                
        if ft_done:
            print('RESTORING FINETUNING WEIGHTS')
            astromer.load_weights(os.path.join(SAVEPATH, 'weights')).expect_partial()
        else:
            print('FINETUNING FROM SCRATCH')
            astromer.fit(trainloader, 
                         epochs=N_EPOCHS, 
                         validation_data=validloader,
                         callbacks=[WandbMetricsLogger(log_freq='epoch'),
                                    EarlyStopping(monitor='val_loss',
                                                  patience = 20,
                                                  restore_best_weights=True),
                                    ModelCheckpoint(filepath=os.path.join(SAVEPATH, 'weights'),
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    save_best_only=True)])

        # ============================================================================
        # =========== CLASSIFICATION==================================================
        # ============================================================================
        CLFSAVEPATH = os.path.join(PTWEIGTHS, 'classification', config.dataset_to_ft, 'fold_'+str(config.fold), config.dataset_to_ft+'_20')
        
        num_cls = pd.read_csv(
                os.path.join(curr_dataset, 'objects.csv')).shape[0]

        wandb.log({"clf": config.clf_name, "ft_dataset": curr_dataset})


        clf_model = create_classifier(astromer, window_size, heads=4, head_dim=64, num_cls=num_cls, train_astromer=False, name=config.clf_name)
        data = load_clf_data(curr_dataset, batch_size, window_size, num_cls, 
                             step='classification', debug=False)

        clf_model, backlog_df = classify(clf_model, data, CLFSAVEPATH, lr=1e-5, model_name=clf_model.name)
        wandb.log(backlog_df)

        with open(os.path.join(CLFSAVEPATH, clf_model.name, 'results.json'), 'w') as fp:
            json.dump(backlog_df, fp)

# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project=MASTER_PROJECT_NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

#7beif1u9
if sys.argv[2] != '0':
    print('using previous id: ', sys.argv[2])
    sweep_id = sys.argv[2]

wandb.agent(sweep_id, function=sweep_train, count=10)

