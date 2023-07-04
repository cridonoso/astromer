import tensorflow as tf
from tensorflow.keras.layers import Dropout 
from typing import Optional

class LoRALayer(tf.Module):
    def __init__(
            self, 
            r: int, #rank
            lora_alpha: int, #learning_rate
            lora_dropout: float, #dropout 
            merge_weights: bool, #boolean flag to merge weights
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpga = lora_alpha
        #dropout optional 
        if lora_dropout > 0.:
            self.lora_dropout = Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        #mark the weight as unmerged
        self.merged = False 
        self.merge_weights = merge_weights


class Embedding(tf.keras.layers.Embedding, LoRALayer):
    #LoRA implemented in dense layer 
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            r: int =0, 
            lora_alpha: int =1, 
            merge_weights: bool = True, 
            **kwargs
    ):
        tf.keras.layers.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init___(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        #Acutal trainable parameters 
        if r > 0:
            self.lora_A = self.add_weight(shape=(r, num_embeddings), trainable=True, initializer='zeros')
            self.lora_B = self.add_weight(shape=(embedding_dim, r), trainable = True, initializer='zeros')
            self.scaling = self.lora_alpha/ self.r
            #freezing the pretrained weight matrix 
            self.embeddings.trainable = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            #initialize A the sme way as the default for tf.keras.layers.Dense and B to zero
            self.lora_A.assign(tf.zeros_like(self.lora_A))
            self.lora_B.assign(tf.random.normal(self.lora_B.shape))

    def train(self, mode: bool =True):
        self.training = mode 
        if mode:
            if self.merge_weights and self.merged:
                #make sure the weights are not merged 
                if self.r > 0:
                    self.embeddings.assign_sub(tf.linalg.matmul(self.lora_B, self.lora_A))
                self.merged = False 
        
        else:
            if self.merge_weights and not self.merged:
                #merge the weights and mark it 
                if self.r > 0:
                    self.embeddings.assign_add(tf.linalg.matmul(self.lora_B, self.lora_A, transpose_a=True) * self.scaling)
                self.merged = True

    def call(self, x):
        if self.r > 0 and not self.merged:
            result = super().call(x)
            if self.r > 0:
                after_A = tf.nn.embedding_lookup(self.lora_A, x)
                result += tf.linalg.matmul(after_A, self.lora_B, transpose_b = True) * self.scaling 
                return result 
        else:
            return super().call(x)
        
