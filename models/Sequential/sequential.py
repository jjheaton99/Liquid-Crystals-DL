from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model

from keras.utils.vis_utils import plot_model

#set up the data generators
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0/255)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    directory='D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/train',
    target_size=(200, 200),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)
n_train = train_gen.n

valid_gen = valid_datagen.flow_from_directory(
    directory='D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/valid',
    target_size=(200, 200),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)
n_valid = valid_gen.n

def plot_loss_acc_history(history):
    fig, axis = plt.subplots(2)
    fig.suptitle('Training losses and accuracies')
    
    axis[0].plot(history.history['loss'], label='loss')
    axis[0].plot(history.history['val_loss'], label='val_loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend(loc='upper right')
    
    axis[1].plot(history.history['accuracy'], label='accuracy')
    axis[1].plot(history.history['val_accuracy'], label='val_accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('accuracy')
    axis[1].legend(loc='upper right')

    plt.show()

INPUT_SHAPE = (200, 200, 1)
NUM_CLASSES = 6

TRAIN_STEP_SIZE = n_train//train_gen.batch_size
VALID_STEP_SIZE = n_valid//valid_gen.batch_size

#callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20)
model_save = ModelCheckpoint('checkpoints', save_best_only=True)
learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=10,
                                           verbose=1,
                                           min_lr=1e-5)
"""
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(1000, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(500, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])
"""
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(400, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(200, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

#model = keras.models.load_model('checkpoints')

model.summary()

plot_model(model, to_file='v2.png', show_shapes=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics='accuracy')

history = model.fit(
          x=train_gen,
          steps_per_epoch=TRAIN_STEP_SIZE,
          epochs=1000,
          verbose=2,
          callbacks=[early_stop, model_save, learning_rate_schedule],
          validation_data=valid_gen,
          validation_steps=VALID_STEP_SIZE)
 
model.evaluate(
    valid_gen,
    steps=VALID_STEP_SIZE,
    verbose=2)

plot_loss_acc_history(history)