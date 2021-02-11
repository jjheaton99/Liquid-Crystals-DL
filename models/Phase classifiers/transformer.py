import numpy as np

import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Flatten

class PatchSplit(Layer):
    def __init__(self, patch_dim, flatten=True):
        super(PatchSplit, self).__init__()

        self.patch_dim = patch_dim
        self.flatten = flatten
        
    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.h = input_shape[1]
        self.w = input_shape[2]
        self.c = input_shape[3]
        
    def call(self, inputs):
        if self.h % self.patch_dim != 0 or self.w % self.patch_dim != 0:
            raise Exception('Input image size is not divisible by patch dimensions!')
            
        rows = self.h // self.patch_dim
        cols = self.w // self.patch_dim
        
        if self.flatten:
            flatten_patches = Flatten()
            flat_outputs = tf.Variable(tf.zeros((self.batch_size, 
                                                 rows * cols, 
                                                 self.patch_dim**2 * self.c)))
        
        outputs = tf.Variable(tf.zeros((self.batch_size, 
                                        rows * cols, 
                                        self.patch_dim, 
                                        self.patch_dim, self.c)))
        
        for batch_idx in range(self.batch_size):
            patch_idx = 0
            for row in range(rows):
                for col in range(cols):
                    slice_row = row * self.patch_dim
                    slice_col = col * self.patch_dim
                    outputs = outputs[batch_idx, patch_idx].assign(
                        inputs[batch_idx,
                        slice_row:slice_row+self.patch_dim, 
                        slice_col:slice_col+self.patch_dim])
                    
                    patch_idx += 1
                    
            if self.flatten:
                flat_outputs = flat_outputs[batch_idx].assign(flatten_patches(outputs[batch_idx]))
        
        if self.flatten:            
            return flat_outputs
        else:
            return outputs

class LinearProjection(Layer):
    def __init__(self, output_dim=32):
        super(LinearProjection, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[2], self.output_dim),
            initializer='random_normal',
            trainable=True
        )
        
    def call(self, inputs):
        return tf.einsum('ijn,nm->ijm', inputs, self.w)
    
inputs = tf.convert_to_tensor(np.reshape(np.arange(5*30*30*1), (5, 30, 30, 1)).astype('float32'))
patch_split = PatchSplit(5)
one = patch_split(inputs)
print(np.shape(one.numpy()))
linear = LinearProjection(10)
two = linear(one)
print(np.shape(two.numpy()))
print(two.numpy()[0,0,0])            

"""
class VisionTransformer():
    def __init__(self):
"""
