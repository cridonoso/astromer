import tensorflow as tf
import pandas as pd
import argparse
import toml
import sys
import os

from src.models.astromer_1 import restore_model, predict, get_embeddings, save_embeddings
from src.data.zero import pretraining_pipeline
from datetime import datetime
from tqdm import tqdm

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        
    astromer, model_config = restore_model(opt.pt_folder)   
    
    objects_df = pd.read_csv(os.path.join(opt.data, 'objects.csv'))
    num_cls = objects_df.shape[0]
    
    pbar = tqdm(['train', 'val', 'test'], total=3)
    for subset in pbar:
        dataset = pretraining_pipeline(os.path.join(opt.data, subset),
                                    batch_size=opt.bs,
                                    window_size=model_config['window_size'],
                                    shuffle=False,
                                    sampling=False,
                                    msk_frac=0.,
                                    rnd_frac=0.,
                                    same_frac=0.,
                                    return_ids=True,
                                    key_format='1')

        cls_label = np.concatenate([np.argmax(y.numpy(), 1) for x, (y, oid) in dataset], axis=0)
        oid_list  = np.concatenate([[oid.numpy()[0].decode()] for x, (y, oid) in dataset], axis=0)
        
        obs_emb = get_emb_1(astromer, dataset, model_config)
        
        save_embeddings({'embedding': obs_emb,
                         'oid': oid_list,
                         'cls_label': cls_label}, 
                         opt.target, '{}'.format(subset)
                        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='-1', type=str, help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--data', default='alcock', type=str, help='Data folder where tf.record files are located')
    parser.add_argument('--pt-folder', default='./results/pretraining*', type=str, help='pretrained model folder')
    parser.add_argument('--target', default='./data/embedding*', type=str, help='folder where embeddings will be saved')
    parser.add_argument('--bs', default=2000, type=int,	help='Batch size')

    opt = parser.parse_args()        
    run(opt)
