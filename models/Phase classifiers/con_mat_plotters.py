import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix

"""
import sys
sys.path.insert(1, 'C:/MPhys project/Liquid-Crystals-DL/misc scripts')
"""

from miscScripts.image_data_transformer import transform_image
#from miscScripts.confusion_matrix_plotter import display_confusion_matrix, display_2_confusion_matrices
from model_training import load_model_fl

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

def create_test_gen(test_dir, binary=False, image_size=256):
        #counts all files in subdirectories in test folder
    NUM_IMAGES = sum(len(files) for _, _, files in os.walk(test_dir))
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    if binary:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'    
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(image_size//1, image_size//1),
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=NUM_IMAGES,
        shuffle=False)
    
    return test_gen

def get_labels_and_preds(model_dir, test_gen, binary=False, sequential=False, 
                         focal_loss=False, evaluate=True): 
    NUM_IMAGES = test_gen.batch_size    
    test_batch = test_gen.next()
    x = test_batch[0]
    y = test_batch[1]

    if binary:
        y_true = y
    else:
        y_true = np.argmax(y, axis=1)
    
    if focal_loss:
        model = load_model_fl(model_dir)
    else:
        model = tf.keras.models.load_model(model_dir)
    
    if evaluate:
        #evaluate for total test set accuracy
        model.evaluate(
            x,
            y,
            batch_size = 1,
            steps=NUM_IMAGES,
            verbose=2)
    
    if binary:
        y_pred = model.predict(x)
        for index in range(NUM_IMAGES):
            if y_pred[index] >= 0:
                y_pred[index] = 1
            else:
                y_pred[index] = 0
        
    elif sequential:
        y_pred = model.predict_classes(x)
        
    else:
        y_pred = np.argmax(model.predict(x), axis=1)

    return y_true, y_pred    

def get_multi_labels_preds(model_dirs, test_gen, binary=False, sequential=False, 
                           focal_loss=False, evaluate=True):
    labels_preds = []
    for model_dir in model_dirs:
        labels_preds.append((get_labels_and_preds(model_dir, test_gen, binary, 
                                                 sequential, focal_loss, evaluate)))
    return labels_preds    

def display_confusion_matrix(model_dir, test_gen, class_names, title='Confusion Matrix', 
                             binary=False, sequential=False, focal_loss=False, evaluate=True, font_scale=1.0):
    y_true, y_pred = get_labels_and_preds(model_dir, test_gen, binary, sequential, focal_loss, evaluate)
    
    con_mat = confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=class_names, columns=class_names)
    
    figsize=(len(class_names), len(class_names))
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True)
    plt.tight_layout()
    plt.ylabel('True phase')
    plt.xlabel('Predicted phase')
    plt.show()
    
def display_mean_confusion_matrix(model_dirs, test_gen, class_names, title='Confusion Matrix', 
                                  sub_title_1='mean', sub_title_2='uncertainty', binary=False, sequential=False, 
                                  focal_loss=False, evaluate=True, figsize=None, font_scale=1.0, reorder_2_phase=False):
    labels_preds = get_multi_labels_preds(model_dirs, test_gen, binary, sequential, 
                           focal_loss, evaluate)
    
    matrix_dim = len(class_names)
    con_mats = np.empty((len(labels_preds), matrix_dim, matrix_dim))
    for index, label_pred in enumerate(labels_preds):
        labels = label_pred[0]
        preds = label_pred[1]
        if reorder_2_phase:
            labels = rearrange_2_phase_labels(labels)
            preds = rearrange_2_phase_labels(preds)
        con_mat = confusion_matrix(labels=labels, predictions=preds).numpy()
        con_mats[index] = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        
    con_mats_mean = np.mean(con_mats, axis=0)
    con_mats_mean_df = pd.DataFrame(con_mats_mean, index=class_names, columns=class_names)
    
    con_mats_err = np.std(con_mats, axis=0)
    con_mats_err_df = pd.DataFrame(con_mats_err, index=class_names, columns=class_names)
    
    if figsize is None:
        figsize=(matrix_dim*2, matrix_dim)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mats_mean_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax1)
    sns.heatmap(con_mats_err_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax2)
    
    ax1.set_title(sub_title_1)
    ax1.set_ylabel('True phase')
    ax1.set_xlabel('Predicted phase')
    ax2.set_title(sub_title_2)
    ax2.set_ylabel('True phase')
    ax2.set_xlabel('Predicted phase')
    
    plt.tight_layout(w_pad=0.0, h_pad=1.5)
    plt.show()

#sort true and predicted labels into correct phase order 
#to display in the confusion matrix
def rearrange_4_phase_labels(labels):
    num_labels = np.shape(labels)[0]
    new_labels = np.empty(num_labels)
    
    for index in range(num_labels):
        if labels[index] == 0:
            new_labels[index] = 2 
        elif labels[index] == 1:
            new_labels[index] = 0
        elif labels[index] == 2:
            new_labels[index] = 1
        elif labels[index] == 3:
            new_labels[index] = 3
    
    return new_labels

def rearrange_2_phase_labels(labels):
    num_labels = np.shape(labels)[0]
    new_labels = np.empty(num_labels)
    
    for index in range(num_labels):
        if labels[index] == 0:
            new_labels[index] = 1 
        elif labels[index] == 1:
            new_labels[index] = 0
    
    return new_labels

test_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticIF/test')

display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 3 layers, 8 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=False)
"""
display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 3 layers, 16 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 3 layers, 32 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 4 layers, 8 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 4 layers, 16 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Sequential 4 layers, 32 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 1 blocks, 2 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 1 blocks, 4 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 1 blocks, 8 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 2 blocks, 2 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 2 blocks, 4 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 2 blocks, 8 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 3 blocks, 2 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 3 blocks, 4 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)

display_mean_confusion_matrix(['checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_a',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_b',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_c',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_d',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_e',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_f',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_g',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_h',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_i',
                               'checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_j'], 
                              test_gen, 
                              ['SmI', 'SmF'],
                              'Inception 3 blocks, 8 starting channels',
                              figsize=(5, 2.5),
                              reorder_2_phase=True)
"""
""" 
def display_2_confusion_matrices(y_true_1, y_pred_1, y_true_2, y_pred_2, class_names, title='Confusion Matrix', 
                                 sub_title_1='', sub_title_2='', figsize=(10, 5), font_scale=1.2):
    con_mat_1 = confusion_matrix(labels=y_true_1, predictions=y_pred_1).numpy()
    con_mat_norm_1 = np.around(con_mat_1.astype('float') / con_mat_1.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df_1 = pd.DataFrame(con_mat_norm_1, index=class_names, columns=class_names)
    
    con_mat_2 = confusion_matrix(labels=y_true_2, predictions=y_pred_2).numpy()
    con_mat_norm_2 = np.around(con_mat_2.astype('float') / con_mat_2.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df_2 = pd.DataFrame(con_mat_norm_2, index=class_names, columns=class_names)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mat_df_1, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax1)
    sns.heatmap(con_mat_df_2, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax2)
    
    ax1.set_title(sub_title_1)
    ax1.set_ylabel('True phase')
    ax1.set_xlabel('Predicted phase')
    ax2.set_title(sub_title_2)
    ax2.set_ylabel('True phase')
    ax2.set_xlabel('Predicted phase')
    
    plt.tight_layout(w_pad=4.0, h_pad=1.5)
    plt.show()
"""