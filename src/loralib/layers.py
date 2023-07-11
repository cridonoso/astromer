import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers 
from tensorflow.keras import constraints 
from tensorflow.python.keras.utils import tf_utils 
from tensorflow.python.ops import embedding_ops

class LoRALayer(tf.Module):
    def __init__(
            self, 
            r: int, #rank
            lora_alpha: int, #learning_rate
            lora_dropout: float, #dropout 
            merge_weights: bool, #boolean flag to merge weights
    ):
        self.r = r 
        self.lora_alpha = lora_alpha 
        
        #optional dropout 
        if lora_dropout > 0.:
            self.lora_dropout = layers.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x : x 
        
        #mark the weights as unmerged 
        self.merged = False 
        self.merge_weights = merge_weights



class Embedding(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, r=0, lora_alpha=1, merge_weights=True, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.merged = False
        self.scaling = tf.sqrt(tf.cast(self.embedding_dim, tf.float32))

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.num_embeddings, self.embedding_dim),
                                          initializer='random_normal',
                                          trainable=False,
                                          name="embeddings")

        if self.r > 0:
            self.lora_A = self.add_weight(shape=(self.num_embeddings, self.r),
                                          initializer='zeros',
                                          trainable=True,
                                          name="lora_A")

            self.lora_B = self.add_weight(shape=(self.r, self.embedding_dim),
                                          initializer='random_normal',
                                          trainable=True,
                                          name="lora_B")

        self.reset_parameters()

    def reset_parameters(self):
        random_normal = tf.random.normal(shape = self.embeddings.shape, mean = 0., stddev=1.)
        self.embeddings.assign(random_normal)
    
        if self.r > 0:
            self.lora_A.assign(tf.zeros(shape=self.lora_A.shape))
            random_normal_B = tf.random.normal(shape=self.lora_B.shape, mean=0., stddev=1.)
            self.lora_B.assign(random_normal_B)

    def call(self, inputs, training=None):
        result = embedding_ops.embedding_lookup(self.embeddings, inputs)
        if self.r > 0:
            after_A = embedding_ops.embedding_lookup(self.lora_A, inputs)
            result += tf.linalg.matmul(after_A, self.lora_B) * self.scaling
            if training:
                self.embeddings.assign_add(tf.linalg.matmul(tf.transpose(after_A), tf.transpose(self.lora_B)) * self.scaling)
        return result




