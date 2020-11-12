import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import sys
sys.path.insert(1, 'D:/MPhys project/Liquid-Crystals-DL/misc scripts')

import numpy as np

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from history_plotter import plot_loss_acc_history

import v1
import v2
import v3

BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
DATASET_NUMBER = 2

#set up the data generators
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
    directory='D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set%d/train' % DATASET_NUMBER,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)
n_train = train_gen.n

valid_gen = valid_datagen.flow_from_directory(
    directory='D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set%d/valid' % DATASET_NUMBER,
    target_size=IMAGE_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True)
n_valid = valid_gen.n

TRAIN_STEP_SIZE = n_train//train_gen.batch_size
VALID_STEP_SIZE = n_valid//valid_gen.batch_size

def train_model(model, model_name, train_gen, valid_gen, save_diagram=False):
    #callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=100)
    model_save = ModelCheckpoint('checkpoints/'+model_name, save_best_only=True)
    learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=20,
                                               verbose=1,
                                               min_lr=1e-5)
    
    model.summary()

    if save_diagram:
        plot_model(model, to_file='plots/architectures/'+model_name+'.png', show_shapes=True)
    
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
    
    best_model = load_model('checkpoints/'+model_name)
    accuracy = best_model.evaluate(
                    valid_gen,
                    steps=VALID_STEP_SIZE,
                    verbose=2)[1]
    
    plot_loss_acc_history(history)
    
    return accuracy
    
accuracies = np.empty(6)

accuracies[0] = train_model(v3.conv_1, 'v3_conv_1', train_gen, valid_gen)
accuracies[1] = train_model(v3.conv_2, 'v3_conv_2', train_gen, valid_gen)
accuracies[2] = train_model(v3.conv_3, 'v3_conv_3', train_gen, valid_gen)
accuracies[3] = train_model(v3.conv_4, 'v3_conv_4', train_gen, valid_gen)
accuracies[4] = train_model(v3.conv_5, 'v3_conv_5', train_gen, valid_gen)
accuracies[5] = train_model(v3.conv_6, 'v3_conv_6', train_gen, valid_gen)

print(accuracies)