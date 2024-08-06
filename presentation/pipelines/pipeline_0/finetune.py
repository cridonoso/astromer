import tensorflow as tf
import argparse
import toml
import sys
import os

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.optimizers import Adam
from presentation.pipelines.steps.model_design import load_pt_model 
from presentation.pipelines.steps.load_data import build_loader 
from presentation.pipelines.steps.metrics import evaluate_ft

def ft_step(opt, data_path):
    factos = data_path.split('/')
    ft_model = '/'.join(factos[-3:])

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    EXPDIR = os.path.join(opt.pt_model, '..', opt.exp_name, ft_model)
    print('[INFO] Exp dir: ', EXPDIR)
    os.makedirs(EXPDIR, exist_ok=True)
    
    
    # ======= MODEL ========================================
    ft_opt = Adam(opt.lr, 
                  beta_1=0.9,
                  beta_2=0.98,
                  epsilon=1e-9,
                  name='astromer_optimizer')
    
    
    model, model_config = load_pt_model(opt.pt_model, optimizer=ft_opt)
    
    
    # ========== DATA ========================================
    loaders = build_loader(data_path, 
                           model_config, 
                           batch_size=opt.bs, 
                           clf_mode=False, 
                           normalize='zero-mean', 
                           sampling=False,
                           repeat=1,
                           return_test=True)

    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(model_config, f)

    cbks = [TensorBoard(log_dir=os.path.join(EXPDIR, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=20),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1)]
    
   
    model.fit(loaders['train'], 
              epochs=1000000, 
              batch_size=opt.bs,
              validation_data=loaders['validation'],
              callbacks=cbks)
    
    ft_metrics = evaluate_ft(model, loaders['test'], model_config, prefix='test_')
    with open(os.path.join(EXPDIR, 'test_metrics.toml'), "w") as toml_file:
        toml.dump(ft_metrics, toml_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='finetuning', type=str,
                    help='Project name')
    parser.add_argument('--pt-model', default='-1', type=str,
                        help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--bs', default=2000, type=int,
                        help='Finetuning batch size')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Finetuning learning rate')

    opt = parser.parse_args()        
    # datapaths = ['./data/precords/catalina/fold_0/catalina']  
    datapaths = [
                 './data/records/alcock/fold_0/alcock_20', 
                 './data/records/alcock/fold_1/alcock_20',
                 './data/records/alcock/fold_2/alcock_20',
                 './data/records/alcock/fold_0/alcock_100', 
                 './data/records/alcock/fold_1/alcock_100',
                 './data/records/alcock/fold_2/alcock_100',
                 './data/records/alcock/fold_0/alcock_500', 
                 './data/records/alcock/fold_1/alcock_500',
                 './data/records/alcock/fold_2/alcock_500',
                 './data/records/atlas/fold_0/atlas_20', 
                 './data/records/atlas/fold_1/atlas_20',
                 './data/records/atlas/fold_2/atlas_20',
                 './data/records/atlas/fold_0/atlas_100', 
                 './data/records/atlas/fold_1/atlas_100',
                 './data/records/atlas/fold_2/atlas_100',
                 './data/records/atlas/fold_0/atlas_500', 
                 './data/records/atlas/fold_1/atlas_500',
                 './data/records/atlas/fold_2/atlas_500'
    ]
    
    for dp in datapaths:
        ft_step(opt, dp)
