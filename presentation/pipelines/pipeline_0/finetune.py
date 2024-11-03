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
from src.training.utils import train


def ft_step(opt):
    factos = opt.data.split('/')
    ft_model = '/'.join(factos[-3:])

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    EXPDIR = os.path.join(opt.pt_model, '..', opt.exp_name, ft_model)
    print('[INFO] Exp dir: ', EXPDIR)
    os.makedirs(EXPDIR, exist_ok=True)
    
    
    # ======= MODEL ========================================
    optimizer = Adam(opt.lr, 
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9,
                     name='astromer_optimizer')
    
    
    model, model_config = load_pt_model(opt.pt_model)
    
    
    # ========== DATA ========================================
    loaders = build_loader(opt.data, 
                           model_config, 
                           batch_size=opt.bs, 
                           clf_mode=False, 
                           normalize='zero-mean', 
                           sampling=False,
                           repeat=1,
                           return_test=True)

    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(model_config, f)

    model = train(model, 
                  optimizer, 
                  train_data=loaders['train'], 
                  validation_data=loaders['validation'], 
                  num_epochs=1000000, 
                  es_patience=20, 
                  test_data=None, 
                  project_folder=EXPDIR)

    
    ft_metrics = evaluate_ft(model, loaders['test'])
    with open(os.path.join(EXPDIR, 'test_metrics.toml'), "w") as toml_file:
        toml.dump(ft_metrics, toml_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock_20', type=str,
                    help='DOWNSTREAM data path')
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

    ft_step(opt)
