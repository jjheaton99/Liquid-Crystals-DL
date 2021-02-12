import numpy as np

import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Flatten

class PatchSplit(Layer):
    #A layer that splits input images into patches of given size
    #Optionally flattens output patches
    def __init__(self, patch_dim):
        super(PatchSplit, self).__init__()
        
        self.trainable = False
        self.patch_dim = patch_dim
        
    def call(self, inputs):
        input_shape = inputs.shape
        batch_size = input_shape[0]
        img_h = input_shape[1]
        img_w = input_shape[2]
        img_d = input_shape[3]
        
        if img_h % self.patch_dim != 0 or img_w % self.patch_dim != 0:
            raise Exception('Input image size is not divisible by patch dimensions!')
        
        return tf.reshape(tf.image.extract_patches(images=inputs, 
                                        sizes=[1, self.patch_dim, self.patch_dim, 1],
                                        strides=[1, self.patch_dim, self.patch_dim, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID'), 
                          [batch_size, 
                           (img_h * img_w) // self.patch_dim**2, 
                           self.patch_dim**2 * img_d])

class LinearProjection(Layer):
    #Applies linear transformation to flattened patches
    #Useful to change dimensions of patch vectors
    #Does not use bias units
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
        x_class_batch = tf.expand_dims(tf.stack([self.x_class for _ in range(self.batch_size)]), axis=1)
        #print(tf.shape(inputs))
        #output adds positional embeddings
        return tf.add(tf.concat([x_class_batch, inputs], axis=1), self.embeddings)
    
"""
class VisionTransformer():
    def __init__(self):
"""
"""
inputs = tf.convert_to_tensor(np.reshape(np.arange(5*30*30*1), (5, 30, 30, 1)).astype('float32'))
one = PatchSplit(5)(inputs)
print(np.shape(one.numpy()))
two = LinearProjection(10)(one)
print(np.shape(two.numpy()))
three = PositionEmbedding()(two)
print(np.shape(three.numpy()))
three = three.numpy()

"""
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32

inputs = keras.Input((256,256,1), batch_size=BATCH_SIZE)
patch_split = PatchSplit(16)(inputs)
linear = LinearProjection(32)(patch_split)
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
    verbose=0,
    validation_data=valid_gen,
    validation_steps=valid_gen.n//valid_gen.batch_size)
