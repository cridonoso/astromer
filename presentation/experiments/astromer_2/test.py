import tensorflow as tf

from src.data import pretraining_pipeline

dataset = pretraining_pipeline('./data/records/alcock/fold_0/alcock/train',
                                250,
                                20,
                                0.5,
                                0.,
                                0.2,
                                True,
                                True,
                                repeat=1,
                                num_cls=None,
                                normalize=True,
                                cache=True)

for batch, y in dataset.unbatch():
    print(y['mask_out'])
    print(batch['mask_in'])
    print( batch['mask_in'] == y['mask_out'])
    break
