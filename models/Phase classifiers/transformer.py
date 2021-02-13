"""
Implementation of image transformer.
Based on "AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
"""

import numpy as np

import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Flatten, LayerNormalization, Dropout, Activation, Concatenate
from keras.activations import softmax
from keras.models import Model

class PatchSplit(Layer):
    #A layer that splits input images into patches of given size
    #Optionally flattens output patches
    def __init__(self, patch_dim):
        super(PatchSplit, self).__init__()
        
        self.trainable = False
        self.patch_dim = patch_dim
        
    def call(self, inputs):
        input_shape = inputs.shape
        img_h = input_shape[1]
        img_w = input_shape[2]
        img_d = input_shape[3]
        
        if img_h % self.patch_dim != 0 or img_w % self.patch_dim != 0:
            raise Exception('Input image size is not divisible by patch dimensions!')
        
        patches = tf.image.extract_patches(images=inputs, 
                                           sizes=[1, self.patch_dim, self.patch_dim, 1],
                                           strides=[1, self.patch_dim, self.patch_dim, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        
        return tf.reshape(patches, 
                          [tf.shape(patches)[0], 
                           (img_h * img_w) // self.patch_dim**2, 
                           self.patch_dim**2 * img_d])

class LinearProjection(Layer):
    #linear projection of a single vector
    def __init__(self, output_dim):
        super(LinearProjection, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[0], self.output_dim),
            initializer='random_normal',
            trainable=True
        )
        
    def call(self, inputs):
        return tf.einsum('n,nm->m', inputs, self.w)

class MultiLinearProjection(Layer):
    #Applies same linear transformation to multiple vectors
    #Useful to change dimensions of patch vectors
    #Does not use bias units
    def __init__(self, output_dim):
        super(MultiLinearProjection, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[2], self.output_dim),
            initializer='random_normal',
            trainable=True)
        
    def call(self, inputs):
        return tf.einsum('ijn,nm->ijm', inputs, self.w)
    
class PositionEmbedding(Layer):
    #Adds learnable class vector as well as position embeddings to
    #flattened patch vectors
    def __init__(self):
        super(PositionEmbedding, self).__init__()

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.n_patches = input_shape[1]
        self.dim = input_shape[2]
        
        self.x_class = self.add_weight(
            name='x_class',
            shape=(self.dim,),
            initializer='random_normal',
            trainable=True)

        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.n_patches + 1, self.dim),
            initializer='random_normal',
            trainable=True)

    def call(self, inputs):
        #add x_class to the start of each batch of patches       
        new_batches = tf.TensorArray(tf.float32, tf.shape(inputs)[0])
        index = 0
        for batch in inputs:
            appended_batch = tf.concat([tf.expand_dims(self.x_class, axis=0), batch], axis=0)
            new_batches = new_batches.write(index, appended_batch)
            index += 1
        appended_inputs = new_batches.stack()
        
        #output adds positional embeddings
        return tf.add(appended_inputs, self.embeddings)
    
class MultiDotProductAttention(Layer):
    def __init__(self):
        super(MultiDotProductAttention, self).__init__()
    
    def call(self, inputs):
        #input shape is (batch_size, heads, 3, num_patches, model_dim) for the q, k, v matrices
        q = inputs[:,:,0]
        k = inputs[:,:,1]
        v = inputs[:,:,2]
        
        return tf.matmul(softmax((tf.matmul(q, k, transpose_b=True) /
                                  tf.sqrt(tf.cast(tf.shape(inputs)[-1], 'float32')))), v)
        
class MultiSelfAttention(Layer):
    def __init__(self, num_heads, head_dim=None):
        super(MultiSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def build(self, input_shape):
        self.d_model = input_shape[2]
            
        head_dim = 0
        if self.head_dim is not None:
            head_dim = self.head_dim
            
        else:
            if self.d_model % self.num_heads != 0:
                raise Exception('d_model is not divisible by num_heads')
            head_dim = self.d_model // self.num_heads
        
        self.w_qkv = self.add_weight(
            name='w_qkv',
            shape=(self.num_heads, 3, self.d_model, head_dim),
            initializer='random_normal',
            trainable=True)
        
        self.w_o = self.add_weight(
            name='w_o',
            shape=(head_dim * self.num_heads, self.d_model),
            initializer='random_normal',
            trainable=True)
        
    def call(self, inputs):
        lin_proj_to_heads = tf.einsum('bpv,havn->bhapn', inputs, self.w_qkv)
        attention = MultiDotProductAttention()(lin_proj_to_heads)
        new_batches = tf.TensorArray(tf.float32, tf.shape(attention)[0])       
        batch_idx = 0
        for batch in attention:
            concat_batch = tf.concat(tf.unstack(batch), axis=-1)  
            new_batches = new_batches.write(batch_idx, concat_batch)
            batch_idx += 1
        concat = new_batches.stack()
        return tf.einsum('bpi,ij->bpj', concat, self.w_o)
 
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
   
def Encoder(inputs, model_dim, num_msa_heads, head_dim=None, dropout_rate=0.0):
    layer_norm_1 = LayerNormalization()(inputs)
    multi_attention = MultiSelfAttention(num_msa_heads, head_dim)(layer_norm_1)
    add = tf.add(inputs, multi_attention)
    layer_norm_2 = LayerNormalization()(add)
    dense_1 = Dense(model_dim)(layer_norm_2)
    gelu_1 = Activation(gelu)(dense_1)
    dropout_1 = Dropout(dropout_rate)(gelu_1)
    dense_2 = Dense(model_dim)(dropout_1)
    gelu_2 = Activation(gelu)(dense_2)
    dropout_2 = Dropout(dropout_rate)(gelu_2)
    return tf.add(add, dropout_2)
    
class ExtractXClass(Layer):
    def __init__(self):
        super(ExtractXClass, self).__init__()
        
    def call(self, inputs):
        return inputs[:,0]
    
def VisionTransformer(input_shape, num_classes, patch_dim, model_dim, num_encoders, 
                 num_msa_heads=8, dropout_rate=0.0):
    inputs = keras.Input(shape=input_shape)
    x = PatchSplit(patch_dim)(inputs)
    x = MultiLinearProjection(model_dim)(x)
    x = PositionEmbedding()(x)
    for _ in range(num_encoders):
        x = Encoder(x, model_dim, num_msa_heads)
    x = ExtractXClass()(x)
    x = Dense(model_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(model_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation=softmax)(x)
    return Model(inputs=inputs, outputs=outputs)