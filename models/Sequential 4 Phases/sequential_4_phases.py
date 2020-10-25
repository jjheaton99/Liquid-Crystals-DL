import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

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
    rotation_range=30,
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
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
    fig.suptitle('Sequential model training losses and accuracies')
    
    axis[0].plot(history.history['loss'], label='loss')
    axis[0].plot(history.history['val_loss'], label='val_loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend(loc='upper right')
    
    axis[1].plot(history.history['accuracy'], label='accuracy')
    axis[1].plot(history.history['val_accuracy'], label='val_accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy')
    axis[1].legend(loc='lower right')

    plt.show()

INPUT_SHAPE = (200, 200, 1)
NUM_CLASSES = 4

TRAIN_STEP_SIZE = n_train//train_gen.batch_size
VALID_STEP_SIZE = n_valid//valid_gen.batch_size

#callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=30)
model_save = ModelCheckpoint('checkpoints/v1', save_best_only=True)
learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=10,
                                           verbose=1,
                                           min_lr=1e-5)

"""
#model v1
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])
"""
#model v2
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

if __name__ == '__main__':
    model.summary()
    
    plot_model(model, to_file='plots/v2.png', show_shapes=True)

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
