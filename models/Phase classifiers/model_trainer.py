import os
from os.path import join
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import sys
sys.path.insert(1, 'C:/MPhys project/Liquid-Crystals-DL/misc scripts')

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from history_plotter import plot_loss_acc_history

import v1_4_phases
import v2_4_phases
import v3_4_phases
import smectic_models
import smecticAC_models

train_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/train'
valid_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/valid'
test_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test'

train_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/train'
valid_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/valid'
test_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/test'

train_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/train'
valid_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/valid'
test_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/test'

def create_generators(train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir, batch_size=32, 
                      image_shape=(256, 256), flip_augs_only=True, binary=False):
    train_datagen = ImageDataGenerator()
    
    if flip_augs_only:
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1.0/255)
        
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            rescale=1.0/255)
    
    valid_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    if binary:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'
    
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True)
    
    valid_gen = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True)
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True)
    
    return train_gen, valid_gen, test_gen

def train_model(model, model_name, train_gen, valid_gen, test_gen, 
                save_dir='checkpoints', binary=False, save_diagram=False):
    #callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=100)
    model_save = ModelCheckpoint(join(save_dir, model_name), save_best_only=True)
    learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=20,
                                               verbose=1,
                                               min_lr=1e-5)
    
    model.summary()

    if save_diagram:
        plot_model(model, to_file='plots/architecture diagrams/'+model_name+'.png', show_shapes=True)
    
    if binary:
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics='accuracy')

    plot_loss_acc_history(model.fit(
                              x=train_gen,
                              steps_per_epoch=train_gen.n//train_gen.batch_size,
                              epochs=1000,
                              verbose=2,
                              callbacks=[early_stop, model_save, learning_rate_schedule],
                              validation_data=valid_gen,
                              validation_steps=valid_gen.n//valid_gen.batch_size))
    
    best_model = load_model(join(save_dir, model_name))
    val_acc = best_model.evaluate(
                    valid_gen,
                    steps=valid_gen.n//valid_gen.batch_size,
                    verbose=2)[1]
    
    test_acc = best_model.evaluate(
                    test_gen,
                    steps=test_gen.n//test_gen.batch_size,
                    verbose=2)[1]
    
    return val_acc, test_acc
    
def evaluate_model(model, valid_gen, test_gen):
    val = model.evaluate(valid_gen,
                          steps=valid_gen.n//valid_gen.batch_size,
                          verbose=2)[1]
    
    test = model.evaluate(test_gen,
                          steps=test_gen.n//test_gen.batch_size,
                          verbose=2)[1]
    
    return val, test

def train_all_v3_models(train_dir, valid_dir, test_dir, result_save_dir):
    train_gen_all_256, valid_gen_all_256, test_gen_all_256 = create_generators(image_shape=(256, 256), 
                                                               flip_augs_only=False)
    
    train_gen_flip_256, valid_gen_flip_256, test_gen_flip_256 = create_generators(image_shape=(256, 256), 
                                                                 flip_augs_only=True)
    
    train_gen_all_128, valid_gen_all_128, test_gen_all_128 = create_generators(image_shape=(128, 128), 
                                                               flip_augs_only=False)
    
    train_gen_flip_128, valid_gen_flip_128, test_gen_flip_128 = create_generators(image_shape=(128, 128), 
                                                                 flip_augs_only=True)
    
    all_256_val = np.empty(6)
    all_256_test = np.empty(6)
    
    flip_256_val = np.empty(6)
    flip_256_test = np.empty(6)
    
    all_128_val = np.empty(6)
    all_128_test = np.empty(6)
    
    flip_128_val = np.empty(6)
    flip_128_test = np.empty(6)

    all_256_val[0], all_256_test[0] = train_model(v3_4_phases.all_256_1,
                                                  'all_256_1',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    all_256_val[1], all_256_test[1] = train_model(v3_4_phases.all_256_2,
                                                  'all_256_2',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    all_256_val[2], all_256_test[2] = train_model(v3_4_phases.all_256_3,
                                                  'all_256_3',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    all_256_val[3], all_256_test[3] = train_model(v3_4_phases.all_256_4,
                                                  'all_256_4',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    all_256_val[4], all_256_test[4] = train_model(v3_4_phases.all_256_5,
                                                  'all_256_5',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    all_256_val[5], all_256_test[5] = train_model(v3_4_phases.all_256_6,
                                                  'all_256_6',
                                                  train_gen_all_256,
                                                  valid_gen_all_256,
                                                  test_gen_all_256)
    
    flip_256_val[0], flip_256_test[0] = train_model(v3_4_phases.flip_256_1,
                                                  'flip_256_1',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)
    flip_256_val[1], flip_256_test[1] = train_model(v3_4_phases.flip_256_2,
                                                  'flip_256_2',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)
    flip_256_val[2], flip_256_test[2] = train_model(v3_4_phases.flip_256_3,
                                                  'flip_256_3',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)
    flip_256_val[3], flip_256_test[3] = train_model(v3_4_phases.flip_256_4,
                                                  'flip_256_4',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)
    flip_256_val[4], flip_256_test[4] = train_model(v3_4_phases.flip_256_5,
                                                  'flip_256_5',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)
    flip_256_val[5], flip_256_test[5] = train_model(v3_4_phases.flip_256_6,
                                                  'flip_256_6',
                                                  train_gen_flip_256,
                                                  valid_gen_flip_256,
                                                  test_gen_flip_256)

    all_128_val[0], all_128_test[0] = train_model(v3_4_phases.all_128_1,
                                                  'all_128_1',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)
    all_128_val[1], all_128_test[1] = train_model(v3_4_phases.all_128_2,
                                                  'all_128_2',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)
    all_128_val[2], all_128_test[2] = train_model(v3_4_phases.all_128_3,
                                                  'all_128_3',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)
    all_128_val[3], all_128_test[3] = train_model(v3_4_phases.all_128_4,
                                                  'all_128_4',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)
    all_128_val[4], all_128_test[4] = train_model(v3_4_phases.all_128_5,
                                                  'all_128_5',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)
    all_128_val[5], all_128_test[5] = train_model(v3_4_phases.all_128_6,
                                                  'all_128_6',
                                                  train_gen_all_128,
                                                  valid_gen_all_128,
                                                  test_gen_all_128)

    flip_128_val[0], flip_128_test[0] = train_model(v3_4_phases.flip_128_1,
                                                  'flip_128_1',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)
    flip_128_val[1], flip_128_test[1] = train_model(v3_4_phases.flip_128_2,
                                                  'flip_128_2',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)
    flip_128_val[2], flip_128_test[2] = train_model(v3_4_phases.flip_128_3,
                                                  'flip_128_3',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)
    flip_128_val[3], flip_128_test[3] = train_model(v3_4_phases.flip_128_4,
                                                  'flip_128_4',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)
    flip_128_val[4], flip_128_test[4] = train_model(v3_4_phases.flip_128_5,
                                                  'flip_128_5',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)
    flip_128_val[5], flip_128_test[5] = train_model(v3_4_phases.flip_128_6,
                                                  'flip_128_6',
                                                  train_gen_flip_128,
                                                  valid_gen_flip_128,
                                                  test_gen_flip_128)   

    accs = np.round(100*np.array([all_256_val,
                                  all_256_test,
                                  flip_256_val,
                                  flip_256_test,
                                  all_128_val,
                                  all_128_test,
                                  flip_128_val,
                                  flip_128_test]), 2)
    
    print(accs)
    
    pd.DataFrame(accs).to_csv(join(result_save_dir, 'accs3.csv'))

train_gen, valid_gen, test_gen = create_generators(train_dir_smecticAC,
                                                   valid_dir_smecticAC,
                                                   test_dir_smecticAC,
                                                   binary=True)

train_model(smecticAC_models.flip_256_inception_2,
            'flip_256_inception_2',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticAC',
            binary=True,
            save_diagram=False)

train_model(smecticAC_models.flip_256_inception_3,
            'flip_256_inception_3',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticAC',
            binary=True,
            save_diagram=False)