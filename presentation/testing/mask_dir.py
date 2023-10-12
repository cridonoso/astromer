import tensorflow as tf 
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# magnitudes = tf.linspace(0., 2., 3)
magnitudes = tf.range(0, 3, dtype=tf.float32)
magnitudes = tf.reshape(magnitudes, [1, 1, 3, 1])

wq = Dense(2, use_bias=False)
wk = Dense(2, use_bias=False)
wv = Dense(2, use_bias=False)

Q = wq(magnitudes) * 0 +magnitudes
K = wq(magnitudes) * 0 +magnitudes
V = wv(magnitudes)


matmul_qk = tf.matmul(Q, K, transpose_b=True) 

# matmul_qk = tf.nn.softmax(matmul_qk, axis=-1, name='MaskedSoftMax') 

a = tf.reduce_sum(matmul_qk[0, 0], axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(10,10))
axes[0].imshow(Q[0, 0], cmap='Blues')

axes[1].imshow(tf.transpose(K[0, 0]), cmap='Blues')
im = axes[2].imshow(matmul_qk[0, 0], cmap='Blues')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.35, 0.05, 0.3])
fig.colorbar(im, cax=cbar_ax)

axes[0].set_title('Q')
axes[1].set_title('K')
axes[2].set_title('QK')

for ax in axes[:-1]:
	ax.axis('off')
axes[-1].set_yticks([0,1,2])

for i in range(2):
	for j in range(3):
		axes[0].text(i-0.1, j-0.1, '{:.2f}'.format(Q[0, 0, j, i]))
		axes[1].text(j-0.1, i-0.1, '{:.2f}'.format(tf.transpose(K[0, 0, j, i])))

for i in range(3):
	for j in range(3):
		axes[2].text(i-0.1, j-0.1, '{:.2f}'.format(matmul_qk[0, 0, j, i]))

fig.savefig('./presentation/figures/testing_mask_dir.png')