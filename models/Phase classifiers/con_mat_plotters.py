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
        target_size=(int(image_size), int(image_size)),
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
                                  sub_title_1='(b)', sub_title_2='(c)', binary=False, sequential=False, 
                                  focal_loss=False, evaluate=True, figsize=None, font_scale=1.0, reorder_2_phase=False,
                                  reorder_ChACHex=False, from_csv=False, csv_path_mean='', csv_path_err='', 
                                  save_csv=False, csv_save_dir='', csv_name='csv'):
    if from_csv:
        con_mats_mean_df = pd.read_csv(csv_path_mean, header=0, names=class_names)
        con_mats_err_df = pd.read_csv(csv_path_err, header=0, names=class_names)
        
    else:
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
            elif reorder_ChACHex:
                labels = rearrange_ChACHex_phase_labels(labels)
                preds = rearrange_ChACHex_phase_labels(preds)
            con_mat = confusion_matrix(labels=labels, predictions=preds).numpy()
            con_mats[index] = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], 2)
            
        con_mats_mean = np.around(np.mean(con_mats, axis=0), 2)
        con_mats_mean_df = pd.DataFrame(con_mats_mean, index=class_names, columns=class_names)
        
        con_mats_err = np.around(np.std(con_mats, axis=0), 2)
        con_mats_err_df = pd.DataFrame(con_mats_err, index=class_names, columns=class_names)
        
        if save_csv:
            con_mats_mean_df.to_csv(csv_save_dir + '/' + csv_name + 'mean.csv')
            con_mats_err_df.to_csv(csv_save_dir + '/' + csv_name + 'err.csv')
    
    if figsize is None:
        figsize=(matrix_dim, matrix_dim)
        
    sns.set(font_scale=font_scale)
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    sns.heatmap(con_mats_mean_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax1)
    
    #ax1.set_title(sub_title_1)
    ax1.set_ylabel('True Phase')
    ax1.set_xlabel('Predicted Phase')
    
    #plt.tight_layout(w_pad=0.0, h_pad=1.5)
    plt.show()
    
    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    sns.heatmap(con_mats_err_df, annot=True, cmap=plt.cm.BuPu, cbar=False, square=True, ax=ax2)
    
    #ax2.set_title(sub_title_2)
    ax2.set_ylabel('True Phase')
    ax2.set_xlabel('Predicted Phase')

    #plt.tight_layout(w_pad=0.0, h_pad=1.5)
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

def rearrange_ChACHex_phase_labels(labels):
    num_labels = np.shape(labels)[0]
    new_labels = np.empty(num_labels)
    
    for index in range(num_labels):
        if labels[index] == 0:
            new_labels[index] = 0 
        elif labels[index] == 1:
            new_labels[index] = 3
        elif labels[index] == 2:
            new_labels[index] = 1
        elif labels[index] == 3:
            new_labels[index] = 2
    
    return new_labels

if __name__ == '__main__':
    csv_path = 'C:/MPhys project/Liquid-Crystals-DL/models/Phase classifiers/plots/con mats/FINAL'
    
    plt.rcParams['axes.titley'] = 1.05
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    test_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChSm/test')

    
    display_mean_confusion_matrix(['checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_a',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_b',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_c',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_d',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_e',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_f',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_g',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_h',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_i',
                                   'checkpoints/ChSm/inception/inc_1_16_batch16_lr1e-4_j'], 
                                  test_gen, 
                                  ['Ch', 'Sm'],
                                  '',
                                  figsize=(4, 4),
                                  font_scale=1.8,
                                  from_csv=True,
                                  csv_path_mean=csv_path+'/inc_ChSm_mean.csv',
                                  csv_path_err=csv_path+'/inc_ChSm_err.csv',
                                  csv_save_dir=csv_path,
                                  csv_name='inc_ChSm_')

    test_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/AC/test')
 
    display_mean_confusion_matrix(['checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_a',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_b',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_c',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_d',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_e',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_f',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_g',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_h',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_i',
                                   'checkpoints/AC/Sem2/inception/inc_2_2_batch16_lr1e-4_j'], 
                                  test_gen, 
                                  ['SmA', 'SmC'],
                                  '',
                                  figsize=(4, 4),
                                  font_scale=1.8,
                                  from_csv=True,
                                  csv_path_mean=csv_path+'/inc_AC_mean.csv',
                                  csv_path_err=csv_path+'/inc_AC_err.csv',
                                  csv_save_dir=csv_path,
                                  csv_name='inc_AC_')
    