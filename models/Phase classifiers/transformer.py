"""
Implementation of image transformer.
Based on "AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
"""

import numpy as np

import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Flatten
from keras.activations import softmax

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
            trainable=True
        )
        
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
            trainable=True
        )

        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.n_patches + 1, self.dim),
            initializer='random_normal',
            trainable=True
        )

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
        if self.head_dim != None:
            head_dim = self.head_dim
            
        else:
            if self.d_model % self.num_heads != 0:
                raise Exception('d_model is not divisible by num_heads')
            head_dim = self.d_model // self.num_heads
        
        self.w_qkv = self.add_weight(
            name='w_qkv',
            shape=(self.num_heads, 3, self.d_model, head_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.w_o = self.add_weight(
            name='w_o',
            shape=(head_dim * self.num_heads, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        
        def call(self, inputs):
            lin_proj_to_heads = tf.einsum('bpv,havn->bhapn', inputs, self.w_qkv)
            attention = MultiDotProductAttention()(lin_proj_to_heads)
            new_batches = tf.TensorArray(tf.float32, tf.shape(attention)[0])
            
            index = 0
            for batch in attention:
                concat_batch = tf.concat([head for head in batch], axis=-1)
                new_batches = new_batches.write(index, concat_batch)
                index += 1
            concat = new_batches.stack()
            
            return tf.einsum('bpi,ij->bpj', concat, self.w_o)
    
"""
class VisionTransformer():
    def __init__(self):
"""
"""
inputs = tf.convert_to_tensor(np.reshape(np.arange(5*30*30*1), (5, 30, 30, 1)).astype('float32'))
one = PatchSplit(5)(inputs)
print(np.shape(one.numpy()))
two = MultiLinearProjection(10)(one)
print(np.shape(two.numpy()))
three = PositionEmbedding()(two)
print(np.shape(three.numpy()))
three = three.numpy()
"""

a = tf.convert_to_tensor(np.random.rand(2,4,10))
print(a)

print(MultiSelfAttention(num_heads=5)(a))

"""
from keras.preprocessing.image import ImageDataGeneratorc

BATCH_SIZE = 32

inputs = keras.Input((256,256,1))
patch_split = PatchSplit(16)(inputs)
linear = MultiLinearProjection(32)(patch_split)
embed = PositionEmbedding()(linear)
flatten = Flatten()(embed)
dense = Dense(32, activation='relu')(flatten)
output = Dense(4)(dense)

model = keras.models.Model(inputs=inputs, outputs=output)

model.summary()

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0/255)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4-phase/train'
valid_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4-phase/valid'

train_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)

valid_gen = valid_datagen.flow_from_directory(
    directory=valid_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics='accuracy')

model.fit(
    x=train_gen,
    steps_per_epoch=train_gen.n//train_gen.batch_size,
    epochs=10,
    verbose=1,
    validation_data=valid_gen,
    validation_steps=valid_gen.n//valid_gen.batch_size)
"""