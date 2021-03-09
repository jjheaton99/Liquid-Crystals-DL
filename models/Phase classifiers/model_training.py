import os
from os.path import join

"""
import sys
sys.path.insert(1, 'C:/MPhys project/Liquid-Crystals-DL/misc scripts')
"""

import numpy as np
import pandas as pd

import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model

import vision_transformer

from miscScripts.history_plotter import plot_loss_acc_history

def create_generators(train_dir, valid_dir, test_dir, batch_size=32, 
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

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        #apply softmax
        y_pred = tf.keras.activations.softmax(y_pred)
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

def load_model_fl(filepath, gamma=2.0, alpha=0.25):
    return load_model(filepath, custom_objects={'focal_loss' : categorical_focal_loss(gamma, alpha)})
    
def train_model(model, model_name, train_gen, valid_gen, test_gen=None, save_dir='checkpoints', 
                learning_rate=0.0001, patience=30, reduce_lr=False, is_vit=False, loss='cce', 
                save_history=True, plot_title=None):
    #callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    model_save = ModelCheckpoint(join(save_dir, model_name), save_best_only=True)
    learning_rate_schedule = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=10,
                                               verbose=1,
                                               min_lr=1e-5)
    
    if reduce_lr:
        callbacks=[early_stop, model_save, learning_rate_schedule]  
    else:
        callbacks=[early_stop, model_save]
        
    if loss == 'bce':
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif loss == 'fl':
        loss = [categorical_focal_loss()]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics='accuracy')

    history = model.fit(
                    x=train_gen,
                    steps_per_epoch=train_gen.n//train_gen.batch_size,
                    epochs=1000,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=valid_gen.n//valid_gen.batch_size)
    
    if plot_title is None:
        plot_title = model_name
    plot_loss_acc_history(history, plot_title)

    if save_history:
        pd.DataFrame.from_dict(history.history).to_csv(join(save_dir, model_name, model_name+'.csv'))
    
    if test_gen is not None:
        if is_vit:
            best_model = vision_transformer.load_vit(join(save_dir, model_name))
        elif loss == 'fl':
            best_model = load_model_fl(join(save_dir, model_name))
        else:
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
"""
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
"""