import tensorflow as tf
from src.models.astromer_0 import scaled_dot_product_attention as dot_prod_a0
from src.layers.attention import scaled_dot_product_attention as dot_prod_a1

import matplotlib.pyplot as plt

def split_heads(x, batch_size, num_heads, depth, name='qkv'):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

num_heads = 1
head_dim  = 5
d_model = num_heads * head_dim
depth   = d_model // num_heads # final dimension

q = tf.random.normal([1, 10, head_dim*num_heads], dtype=tf.float32)
k = tf.random.normal([1, 10, head_dim*num_heads], dtype=tf.float32)
v = tf.random.normal([1, 10, head_dim*num_heads], dtype=tf.float32)
q = split_heads(q, batch_size=1, num_heads=num_heads, depth=depth)
k = split_heads(k, batch_size=1, num_heads=num_heads, depth=depth)
v = split_heads(v, batch_size=1, num_heads=num_heads, depth=depth)

mask = tf.convert_to_tensor([[1, 0, 0, 0, 0, 0 ,0, 0, 1, 1]], dtype=tf.float32)
mask = tf.expand_dims(mask, axis=-1)


out_0, w_0 = dot_prod_a0(q, k, v, mask=mask)
out_1, w_1 = dot_prod_a1(q, k, v, mask=mask, mask_format='first')

fig, axes = plt.subplots(1, 3)
axes[0].imshow(w_0[0, 0])
im = axes[1].imshow(w_1[0, 0])
axes[2].imshow(tf.abs(w_1[0, 0] - w_0[0, 0]))
for ax in axes:
	ax.axis('off')

axes[0].set_title('Astromer 0')
axes[1].set_title('Astromer 1')
axes[2].set_title('Residuals')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.35, 0.05, 0.3])
fig.colorbar(im, cax=cbar_ax)

fig.savefig('./presentation/figures/testing_att.png')