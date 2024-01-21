import tensorflow as tf

from src.data.zero import mask_sample
from src.data.masking import get_probed, add_random


lenght = 10
light_curve = tf.convert_to_tensor([tf.range(0, lenght), 
								    tf.range(0, lenght), 
								    tf.range(0, lenght)], dtype=tf.float32)
light_curve = tf.transpose(light_curve)

msk_frac = 0.5
rnd_frac = 0.2

# ==== ASTROMER ZERO ====
input_a0   = {'input': light_curve}
output_0   = mask_sample(input_a0, msk_frac=msk_frac, rnd_frac=rnd_frac, same_frac=rnd_frac, max_obs=10)
N_0_att    = tf.reduce_sum(output_0['mask_in'])
N_0_probed = tf.reduce_sum(output_0['mask_out'])

# === ASTROMER 1 ====
input_a1  = {'input': tf.expand_dims(light_curve, axis=0), 
			 'mask': tf.ones([1, lenght], dtype=tf.float32)}

out_probed = get_probed(input_a1, probed=msk_frac, njobs=2)
out_random = add_random(out_probed, random_frac=rnd_frac, njobs=2)
N_1_att    = tf.reduce_sum(out_random['att_mask'])
N_1_probed = tf.reduce_sum(out_random['probed_mask'])

assert N_0_att == N_1_att
assert N_0_probed == N_1_probed




