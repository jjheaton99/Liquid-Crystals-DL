import os
from os.path import join
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import sys
sys.path.insert(1, 'D:/MPhys project/Liquid-Crystals-DL/misc scripts')

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from history_plotter import plot_loss_acc_history

import v1
import v2
import v3
import inception

train_dir = 'D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/train'
valid_dir = 'D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/valid'
test_dir = 'D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test'

def create_generators(train_dir, valid_dir, test_dir, batch_size=32, 
                      image_shape=(256, 256), flip_augs_only=True):
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
    
    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
    
    valid_gen = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
    
    return train_gen, valid_gen, test_gen

def train_model(model, model_name, train_gen, valid_gen, test_gen, save_diagram=False):
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

    plot_loss_acc_history(model.fit(
                              x=train_gen,
                              steps_per_epoch=train_gen.n//train_gen.batch_size,
                              epochs=1000,
                              verbose=1,
                              callbacks=[early_stop, model_save, learning_rate_schedule],
                              validation_data=valid_gen,
                              validation_steps=valid_gen.n//valid_gen.batch_size))
    
    best_model = load_model('checkpoints/'+model_name)
    valid_acc = best_model.evaluate(
                    valid_gen,
                    steps=valid_gen.n//valid_gen.batch_size,
                    verbose=2)[1]
    
    test_acc = best_model.evaluate(
                    test_gen,
                    steps=test_gen.n//test_gen.batch_size,
                    verbose=2)[1]
    
    return valid_acc, test_acc

def multi_train_all_models(train_dir, valid_dir, test_dir, result_save_dir, num_repeats=3):
    dataframe_cols = ['val1', 'test1', 
                      'val2', 'test2', 
                      'val3', 'test3', 
                      'val4', 'test4', 
                      'val5', 'test5', 
                      'val6', 'test6']
    
    all_256_accs = np.empty((12, num_repeats))
    flip_256_accs = np.empty((12, num_repeats))
    all_128_accs = np.empty((12, num_repeats))
    flip_128_accs = np.empty((12, num_repeats))
    
    def get_accs(model, model_name, image_shape, flip_augs_only):
        train_gen, valid_gen, test_gen = create_generators(train_dir, 
                                                           valid_dir,
                                                           test_dir,
                                                           image_shape=image_shape,
                                                           flip_augs_only=flip_augs_only)
        
        val_accs = test_accs = np.empty(num_repeats)
        for repeat in range(num_repeats):
            val_accs[repeat], test_accs[repeat] = train_model(model,
                                                              model_name,
                                                              train_gen,
                                                              valid_gen,
                                                              test_gen)
        
        return val_accs, test_accs
            
    all_256_accs[0], all_256_accs[1] = get_accs(v3.conv_1_in_256, 
                                                'conv_1_all_256', 
                                                (256, 256), 
                                                False)
    all_256_accs[2], all_256_accs[3] = get_accs(v3.conv_2_in_256, 
                                                'conv_2_all_256', 
                                                (256, 256), 
                                                False)
    all_256_accs[4], all_256_accs[5] = get_accs(v3.conv_3_in_256, 
                                                'conv_3_all_256', 
                                                (256, 256), 
                                                False)
    all_256_accs[6], all_256_accs[7] = get_accs(v3.conv_4_in_256, 
                                                'conv_4_all_256', 
                                                (256, 256), 
                                                False)
    all_256_accs[8], all_256_accs[9] = get_accs(v3.conv_5_in_256, 
                                                'conv_5_all_256', 
                                                (256, 256), 
                                                False)
    all_256_accs[10], all_256_accs[11] = get_accs(v3.conv_6_in_256, 
                                                'conv_6_all_256', 
                                                (256, 256), 
                                                False)
    
    flip_256_accs[0], flip_256_accs[1] = get_accs(v3.conv_1_in_256, 
                                                'conv_1_flip_256', 
                                                (256, 256), 
                                                True)
    flip_256_accs[2], flip_256_accs[3] = get_accs(v3.conv_2_in_256, 
                                                'conv_2_flip_256', 
                                                (256, 256), 
                                                True)
    flip_256_accs[4], flip_256_accs[5] = get_accs(v3.conv_3_in_256, 
                                                'conv_3_flip_256', 
                                                (256, 256), 
                                                True)
    flip_256_accs[6], flip_256_accs[7] = get_accs(v3.conv_4_in_256, 
                                                'conv_4_flip_256', 
                                                (256, 256), 
                                                True)
    flip_256_accs[8], flip_256_accs[9] = get_accs(v3.conv_5_in_256, 
                                                'conv_5_flip_256', 
                                                (256, 256), 
                                                True)
    flip_256_accs[10], flip_256_accs[11] = get_accs(v3.conv_6_in_256, 
                                                'conv_6_flip_256', 
                                                (256, 256), 
                                                True)
    
    all_128_accs[0], all_128_accs[1] = get_accs(v3.conv_1_in_128, 
                                                'conv_1_all_128', 
                                                (128, 128), 
                                                False)
    all_128_accs[2], all_128_accs[3] = get_accs(v3.conv_2_in_128, 
                                                'conv_2_all_128', 
                                                (128, 128), 
                                                False)
    all_128_accs[4], all_128_accs[5] = get_accs(v3.conv_3_in_128, 
                                                'conv_3_all_128', 
                                                (128, 128), 
                                                False)
    all_128_accs[6], all_128_accs[7] = get_accs(v3.conv_4_in_128, 
                                                'conv_4_all_128', 
                                                (128, 128), 
                                                False)
    all_128_accs[8], all_128_accs[9] = get_accs(v3.conv_5_in_128, 
                                                'conv_5_all_128', 
                                                (128, 128), 
                                                False)
    all_128_accs[10], all_128_accs[11] = get_accs(v3.conv_6_in_128, 
                                                'conv_6_all_128', 
                                                (128, 128), 
                                                False)
    
    flip_128_accs[0], flip_128_accs[1] = get_accs(v3.conv_1_in_128, 
                                                'conv_1_flip_128', 
                                                (128, 128), 
                                                True)
    flip_128_accs[2], flip_128_accs[3] = get_accs(v3.conv_2_in_128, 
                                                'conv_2_flip_128', 
                                                (128, 128), 
                                                True)
    flip_128_accs[4], flip_128_accs[5] = get_accs(v3.conv_3_in_128, 
                                                'conv_3_flip_128', 
                                                (128, 128), 
                                                True)
    flip_128_accs[6], flip_128_accs[7] = get_accs(v3.conv_4_in_128, 
                                                'conv_4_flip_128', 
                                                (128, 128), 
                                                True)
    flip_128_accs[8], flip_128_accs[9] = get_accs(v3.conv_5_in_128, 
                                                'conv_5_flip_128', 
                                                (128, 128), 
                                                True)
    flip_128_accs[10], flip_128_accs[11] = get_accs(v3.conv_6_in_128, 
                                                'conv_6_flip_128', 
                                                (128, 128), 
                                                True)
    
    pd.DataFrame(np.transpose(all_256_accs)).to_csv(join(result_save_dir, 'all_256_accs.csv'), 
                                                    header=dataframe_cols)
    pd.DataFrame(np.transpose(flip_256_accs)).to_csv(join(result_save_dir, 'flip_256_accs.csv'), 
                                                     header=dataframe_cols)
    pd.DataFrame(np.transpose(all_128_accs)).to_csv(join(result_save_dir, 'all_128_accs.csv'), 
                                                    header=dataframe_cols)
    pd.DataFrame(np.transpose(flip_128_accs)).to_csv(join(result_save_dir, 'flip_128_accs.csv'), 
                                                     header=dataframe_cols)

multi_train_all_models(train_dir, 
                       valid_dir, 
                       test_dir, 
                       'D:/MPhys project/Liquid-Crystals-DL/models/4 Phases/multi train results',
                       num_repeats=1)