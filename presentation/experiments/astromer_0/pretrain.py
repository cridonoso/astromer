import tensorflow as tf 
import tomli
import os

from tensorflow.keras.optimizers import Adam
from src.data import pretraining_pipeline
from src.models import get_ASTROMER

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


exp_folder = './weights/astromer_0'
with open(os.path.join(exp_folder, 'config.toml'), mode="rb") as fp:
	config = tomli.load(fp)

# ==============================================================================
# =========================== MODEL ============================================
# ==============================================================================
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

astromer.load_weights(os.path.join(exp_folder, 'weights')).expect_partial()


# ==============================================================================
# ============================ DATA ============================================
# ==============================================================================
BATCH_SIZE = 32
test_batches = pretraining_pipeline(os.path.join(config['pretraining']['data']['path'], 'test'),
                                    BATCH_SIZE,
                                    config['astromer']['window_size'],
                                    config['masking']['mask_frac'],
                                    config['masking']['rnd_frac'],
                                    config['masking']['same_frac'],
                                    config['pretraining']['data']['sampling'],
                                    config['pretraining']['data']['shuffle_test'],
                                    repeat=config['pretraining']['data']['repeat'],
                                    normalize=config['pretraining']['data']['normalize'],
                                    cache=config['pretraining']['data']['cache_test'])

astromer.compile(optimizer=Adam(config['pretraining']['lr']))
astromer.evaluate(test_batches)