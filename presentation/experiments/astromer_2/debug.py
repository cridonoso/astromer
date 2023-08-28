from src.layers.downstream import AstromerEmbedding, ReduceAttention
from src.data.loaders import load_light_curves
import os

pw = './presentation/experiments/astromer_2/results/nsp_cond/2_4_64_rmse_0.5/pretraining'


test_batches = load_light_curves(os.path.join('./data/records/macho_clean', 'test'), 
	                              num_cls=1,
			                      batch_size=16, 
			                      window_size=200, 
			                      repeat=1,
			                      cache=True, 
			                      njobs=None)


layer = AstromerEmbedding(pretrain_weights=pw)

layer.evaluate_on_dataset(test_batches)