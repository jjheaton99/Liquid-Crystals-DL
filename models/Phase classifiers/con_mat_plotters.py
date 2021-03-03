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

def get_labels_and_preds(test_gen, model_dir, binary=False, sequential=False, focal_loss=False, evaluate=True):
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

def get_multi_labels_preds(test_gen, model_dirs, binary=False, sequential=False, focal_loss=False, evaluate=True):
    labels_preds = []
    for model_dir in model_dirs:
        labels_preds.append((get_labels_and_preds(test_gen, model_dir, binary, 
                                                 sequential, focal_loss, evaluate)))
    return labels_preds    

def display_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', font_scale=1.0):
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
    
def display_mean_confusion_matrix(labels_preds, class_names, title='Confusion Matrix', 
                                  sub_title_1='mean', sub_title_2='uncertainty', font_scale=1.0):
    matrix_dim = len(class_names)
    con_mats = np.empty((len(labels_preds), matrix_dim, matrix_dim))
    for index, label_pred in enumerate(labels_preds):
        con_mat = confusion_matrix(labels=label_pred[0], predictions=label_pred[1]).numpy()
        con_mats[index] = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        
    con_mats_mean = np.mean(con_mats, axis=0)
    con_mats_mean_df = pd.DataFrame(con_mats_mean, index=class_names, columns=class_names)
    
    con_mats_err = np.std(con_mats, axis=0)
    con_mats_err_df = pd.DataFrame(con_mats_err, index=class_names, columns=class_names)
    
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
    
    plt.tight_layout(w_pad=4.0, h_pad=1.5)
    plt.show()
    
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

def con_mat_4_phases(test_dir, model_name, title='Confusion Matrix', binary=False, 
                     sequential=False, image_size=256, evaluate=True, font_scale=1.2):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, binary, sequential, image_size, evaluate)
    
    y_true = rearrange_4_phase_labels(y_true)
    y_pred = rearrange_4_phase_labels(y_pred)
    
    class_names = ['Iso', 'N', 'N*', 'Sm']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title,
                             font_scale=font_scale)
    
def con_mat_4_phases_2(test_dir, model_name_1, model_name_2, title='Confusion Matrix', sub_title_1='', sub_title_2='', 
                       sequential_1=True, sequential_2=False, image_size_1=256, image_size_2=256, evaluate=True, 
                       figsize=(10, 5), font_scale=1.2):
    y_true_1, y_pred_1 = get_labels_and_preds(test_dir, model_name_1, False, sequential_1, image_size_1, evaluate)
    
    y_true_1 = rearrange_4_phase_labels(y_true_1)
    y_pred_1 = rearrange_4_phase_labels(y_pred_1)
    
    y_true_2, y_pred_2 = get_labels_and_preds(test_dir, model_name_2, False, sequential_2, image_size_2, evaluate)
    
    y_true_2 = rearrange_4_phase_labels(y_true_2)
    y_pred_2 = rearrange_4_phase_labels(y_pred_2)
    
    class_names = ['Iso', 'N', 'N*', 'Sm']
    
    display_2_confusion_matrices(y_true_1, 
                                 y_pred_1,
                                 y_true_2, 
                                 y_pred_2,
                                 class_names, 
                                 title=title,
                                 sub_title_1=sub_title_1,
                                 sub_title_2=sub_title_2,
                                 figsize=figsize,
                                 font_scale=font_scale)
    
def con_mat_smectic(test_dir, model_name, title, sequential=False, image_size=256, evaluate=True, font_scale=1.2):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, False, sequential, image_size, evaluate)
    
    class_names = ['FSm', 'HSm', 'SC']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title,
                             font_scale=font_scale)
    
def con_mat_smecticAC(test_dir, model_name, title, image_size=256, evaluate=True, font_scale=1.2):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, True, False, image_size, evaluate)
    
    class_names = ['A', 'C']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title,
                             font_scale=font_scale)
    
def con_mat_smecticAC_2(test_dir, model_name_1, model_name_2, title, sub_title_1, sub_title_2, 
                        image_size=256, evaluate=True, figsize=(10, 5), font_scale=1.2):
    y_true_1, y_pred_1 = get_labels_and_preds(test_dir, model_name_1, True, False, image_size, evaluate)
    y_true_2, y_pred_2 = get_labels_and_preds(test_dir, model_name_2, True, False, image_size, evaluate)
    
    class_names = ['SmA', 'SmC']
    
    display_2_confusion_matrices(y_true_1, 
                                 y_pred_1,
                                 y_true_2, 
                                 y_pred_2, 
                                 class_names, 
                                 title=title,
                                 sub_title_1=sub_title_1,
                                 sub_title_2=sub_title_2,
                                 figsize=figsize,
                                 font_scale=font_scale)

test_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/test')

labels_preds_1 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/inc_1a',
                                       'checkpoints/smectic3/sem2/inc_1b',
                                       'checkpoints/smectic3/sem2/inc_1c',
                                       'checkpoints/smectic3/sem2/inc_1d',
                                       'checkpoints/smectic3/sem2/inc_1e',
                                       'checkpoints/smectic3/sem2/inc_1f',
                                       'checkpoints/smectic3/sem2/inc_1g',
                                       'checkpoints/smectic3/sem2/inc_1h',
                                       'checkpoints/smectic3/sem2/inc_1i',
                                       'checkpoints/smectic3/sem2/inc_1j',
                                       'checkpoints/smectic3/sem2/inc_1k',
                                       'checkpoints/smectic3/sem2/inc_1l'])

labels_preds_2 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/inc_2a',
                                       'checkpoints/smectic3/sem2/inc_2b',
                                       'checkpoints/smectic3/sem2/inc_2c',
                                       'checkpoints/smectic3/sem2/inc_2d',
                                       'checkpoints/smectic3/sem2/inc_2e',
                                       'checkpoints/smectic3/sem2/inc_2f',
                                       'checkpoints/smectic3/sem2/inc_2g',
                                       'checkpoints/smectic3/sem2/inc_2h',
                                       'checkpoints/smectic3/sem2/inc_2i',
                                       'checkpoints/smectic3/sem2/inc_2j',
                                       'checkpoints/smectic3/sem2/inc_2k',
                                       'checkpoints/smectic3/sem2/inc_2l'])

labels_preds_3 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/inc_3a',
                                       'checkpoints/smectic3/sem2/inc_3b',
                                       'checkpoints/smectic3/sem2/inc_3c',
                                       'checkpoints/smectic3/sem2/inc_3d',
                                       'checkpoints/smectic3/sem2/inc_3e',
                                       'checkpoints/smectic3/sem2/inc_3f',
                                       'checkpoints/smectic3/sem2/inc_3g',
                                       'checkpoints/smectic3/sem2/inc_3h',
                                       'checkpoints/smectic3/sem2/inc_3i',
                                       'checkpoints/smectic3/sem2/inc_3j',
                                       'checkpoints/smectic3/sem2/inc_3k',
                                       'checkpoints/smectic3/sem2/inc_3l'])

labels_preds_fl_1 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/fl_inc_1a',
                                       'checkpoints/smectic3/sem2/fl_inc_1b',
                                       'checkpoints/smectic3/sem2/fl_inc_1c',
                                       'checkpoints/smectic3/sem2/fl_inc_1d',
                                       'checkpoints/smectic3/sem2/fl_inc_1e',
                                       'checkpoints/smectic3/sem2/fl_inc_1f',
                                       'checkpoints/smectic3/sem2/fl_inc_1g',
                                       'checkpoints/smectic3/sem2/fl_inc_1h',
                                       'checkpoints/smectic3/sem2/fl_inc_1i',
                                       'checkpoints/smectic3/sem2/fl_inc_1j',
                                       'checkpoints/smectic3/sem2/fl_inc_1k',
                                       'checkpoints/smectic3/sem2/fl_inc_1l'],
                                       focal_loss=True)

labels_preds_fl_2 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/fl_inc_2a',
                                       'checkpoints/smectic3/sem2/fl_inc_2b',
                                       'checkpoints/smectic3/sem2/fl_inc_2c',
                                       'checkpoints/smectic3/sem2/fl_inc_2d',
                                       'checkpoints/smectic3/sem2/fl_inc_2e',
                                       'checkpoints/smectic3/sem2/fl_inc_2f',
                                       'checkpoints/smectic3/sem2/fl_inc_2g',
                                       'checkpoints/smectic3/sem2/fl_inc_2h',
                                       'checkpoints/smectic3/sem2/fl_inc_2i',
                                       'checkpoints/smectic3/sem2/fl_inc_2j',
                                       'checkpoints/smectic3/sem2/fl_inc_2k',
                                       'checkpoints/smectic3/sem2/fl_inc_2l'],
                                       focal_loss=True)

labels_preds_fl_3 = get_multi_labels_preds(test_gen, 
                                      ['checkpoints/smectic3/sem2/fl_inc_3a',
                                       'checkpoints/smectic3/sem2/fl_inc_3b',
                                       'checkpoints/smectic3/sem2/fl_inc_3c',
                                       'checkpoints/smectic3/sem2/fl_inc_3d',
                                       'checkpoints/smectic3/sem2/fl_inc_3e',
                                       'checkpoints/smectic3/sem2/fl_inc_3f',
                                       'checkpoints/smectic3/sem2/fl_inc_3g',
                                       'checkpoints/smectic3/sem2/fl_inc_3h',
                                       'checkpoints/smectic3/sem2/fl_inc_3i',
                                       'checkpoints/smectic3/sem2/fl_inc_3j',
                                       'checkpoints/smectic3/sem2/fl_inc_3k',
                                       'checkpoints/smectic3/sem2/fl_inc_3l'],
                                       focal_loss=True)

display_mean_confusion_matrix(labels_preds_1, ['FSm', 'HSm', 'SC'])
display_mean_confusion_matrix(labels_preds_2, ['FSm', 'HSm', 'SC'])
display_mean_confusion_matrix(labels_preds_3, ['FSm', 'HSm', 'SC'])
display_mean_confusion_matrix(labels_preds_fl_1, ['FSm', 'HSm', 'SC'])
display_mean_confusion_matrix(labels_preds_fl_2, ['FSm', 'HSm', 'SC'])
display_mean_confusion_matrix(labels_preds_fl_3, ['FSm', 'HSm', 'SC'])

"""
#outputs prediction for image file and associated confidence
def predict_image(filename, model=model, show=False):
    if show:
        transform_image(Image.open(filename), size=IMAGE_SIZE).show()
    image = transform_image(Image.open(filename), 
                            as_array=True, 
                            size=IMAGE_SIZE,
                            black_and_white=True)/255.0
    #expand to 4D tensor so it fits the batch shape
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image, steps=1, verbose=1)
    pred_tensor = tf.constant(prediction)
    probs = tf.keras.activations.softmax(pred_tensor).numpy()
    pred_class = np.argmax(probs, axis=1)
    
    print('Predicted phase:')
    
    if pred_class[0] == 0:
        print('cholesteric')
    elif pred_class[0] == 1:
        print('isotropic')
    elif pred_class[0] == 2:
        print('nematic')
    elif pred_class[0] == 3:
        print('smectic')
        
    print('Confidence level:')
    print(probs[0][pred_class[0]])
        
    return (pred_class[0], probs[0][pred_class[0]])

predict_image('random tests/test_image_nematic.jpg')
print('actual phase: nematic')
predict_image('random tests/test_image_cholesteric.jpg')
print('actual phase: cholesteric')
predict_image('random tests/test_image_smectic.jpg')
print('actual phase: smectic')
"""