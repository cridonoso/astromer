import unittest
import tensorflow as tf

from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

class TestStringMethods(unittest.TestCase):

    def test_model_save(self):
        data = './data/records/macho/test'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)

        model = ASTROMER()

        w_path = './weights/macho_10022021/'

        # model.compile(optimizer='adam',
        #               loss_rec=custom_rmse,
        #               metric_rec=custom_r2)
        # model.load_weights(w_path)
        # model.evaluate(dataset.take(10))

        reader = tf.train.load_checkpoint(w_path)

        shape_from_key = reader.get_variable_to_shape_map()

        # ====== FIRST LAYER =====
        for l_index in range(2):
            bias_0   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/ffn/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            kernel_0 = reader.get_tensor('layer_with_weights-0/enc_layers/{}/ffn/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            bias_1   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/ffn/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            kernel_1 = reader.get_tensor('layer_with_weights-0/enc_layers/{}/ffn/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))

            beta_0  = reader.get_tensor('layer_with_weights-0/enc_layers/{}/layernorm1/beta/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            gamma_0 = reader.get_tensor('layer_with_weights-0/enc_layers/{}/layernorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            beta_1  = reader.get_tensor('layer_with_weights-0/enc_layers/{}/layernorm2/beta/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            gamma_1 = reader.get_tensor('layer_with_weights-0/enc_layers/{}/layernorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))

            mha_bias   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/dense/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            mha_kernel = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))


            first_fnn = model.get_layer('encoder').enc_layers[l_index].ffn
            first_fnn.set_weights([
                kernel_0, bias_0, kernel_1, bias_1
            ])

            layernorm1 = model.get_layer('encoder').enc_layers[l_index].layernorm1
            layernorm2 = model.get_layer('encoder').enc_layers[l_index].layernorm2
            layernorm1.set_weights([gamma_0, beta_0])
            layernorm2.set_weights([gamma_1, beta_1])

            mixer_dense_0 = model.get_layer('encoder').enc_layers[l_index].mha.dense
            mixer_dense_0.set_weights([mha_kernel, mha_bias])

            wk_0_bias   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wk/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            wk_0_kernel = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wk/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            wq_0_bias   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            wq_0_kernel = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            wv_0_bias   = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wv/bias/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))
            wv_0_kernel = reader.get_tensor('layer_with_weights-0/enc_layers/{}/mha/wv/kernel/.ATTRIBUTES/VARIABLE_VALUE'.format(l_index))

            wk_0 = model.get_layer('encoder').enc_layers[l_index].mha.wk
            wk_0.set_weights([wk_0_kernel, wk_0_bias])

            wq_0 = model.get_layer('encoder').enc_layers[l_index].mha.wq
            wq_0.set_weights([wq_0_kernel, wq_0_bias])

            wv_0 = model.get_layer('encoder').enc_layers[l_index].mha.wv
            wv_0.set_weights([wv_0_kernel, wv_0_bias])



        inp_t_bias   = reader.get_tensor('layer_with_weights-0/inp_transform/bias/.ATTRIBUTES/VARIABLE_VALUE')
        inp_t_kernel = reader.get_tensor('layer_with_weights-0/inp_transform/kernel/.ATTRIBUTES/VARIABLE_VALUE')

        bn_0_beta  = reader.get_tensor('layer_with_weights-1/bn_0/beta/.ATTRIBUTES/VARIABLE_VALUE')
        bn_0_gamma = reader.get_tensor('layer_with_weights-1/bn_0/gamma/.ATTRIBUTES/VARIABLE_VALUE')

        reg_layer_bias   = reader.get_tensor('layer_with_weights-1/reg_layer/bias/.ATTRIBUTES/VARIABLE_VALUE')
        reg_layer_kernel = reader.get_tensor('layer_with_weights-1/reg_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE')

        inp_transform = model.get_layer('encoder').inp_transform
        inp_transform.set_weights([inp_t_kernel, inp_t_bias])

        reg_layer = model.get_layer('regression').reg_layer
        bn_0 = model.get_layer('regression').bn_0
        reg_layer.set_weights([reg_layer_kernel, reg_layer_bias])
        bn_0.set_weights([bn_0_gamma, bn_0_beta])

        import pickle
        with open('weight.pkl', 'rb') as h:
            w = pickle.load(h)

        optimizer = tf.keras.optimizers.Adam(1e-3)

        grad_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        # Apply gradients which don't do nothing with Adam
        optimizer.apply_gradients(zip(zero_grads, grad_vars))
        optimizer.set_weights(w)

        model.compile(optimizer=optimizer,
                      loss_rec=custom_rmse,
                      metric_rec=custom_r2)

        model.fit(dataset.take(10), epochs=5)

        # model.save_weights('./weights/macho_10022021/weights_old.h5')

if __name__ == '__main__':
    unittest.main()
