import unittest
import tensorflow as tf

from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

class TestStringMethods(unittest.TestCase):

    def test_model_save(self):
        data = './data/records/alcock/fold_0/alcock_20/train'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)

        model = ASTROMER()

        w_path = './weights/macho_10022021/weights'

        reader = tf.train.load_checkpoint(w_path)

        shape_from_key = reader.get_variable_to_shape_map()


        bias_0 = reader.get_tensor('layer_with_weights-0/enc_layers/0/ffn/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')
        kernel_0 = reader.get_tensor('layer_with_weights-0/enc_layers/0/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE')
        bias_1 = reader.get_tensor('layer_with_weights-0/enc_layers/0/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')
        kernel_1 = reader.get_tensor('layer_with_weights-0/enc_layers/0/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')

        print(bias_0.shape)
        print(kernel_0.shape)
        print(bias_1.shape)
        print(kernel_1.shape)
        dense_layer_0 = model.get_layer('encoder')

        print('----'*10)
        for k in dense_layer_0.inp_transform.variables:
            print(k.shape)
        #
        # dense_layer_0.inp_transform.set_weights([kernel_0, bias_0])
        #
        # print(dense_layer_0.inp_transform.weights)

        print('----'*10)
        for v in sorted(shape_from_key.keys()):
            print(v)

        # #
        # for v in model.variables:
        #     print(v.name)

        # a = reader.get_tensor('layer_with_weights-0/enc_layers/0/ffn/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')

        # model.load_weights()
        # model.compile(optimizer='adam',
        #               loss_rec=custom_rmse,
        #               metric_rec=custom_r2)
        # model.evaluate(dataset)






if __name__ == '__main__':
    unittest.main()
