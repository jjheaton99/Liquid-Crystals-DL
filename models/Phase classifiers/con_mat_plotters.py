import numpy as np
import os

import sys
sys.path.insert(1, 'C:/MPhys project/Liquid-Crystals-DL/misc scripts')

from image_data_transformer import transform_image
from confusion_matrix_plotter import display_confusion_matrix, display_2_confusion_matrices

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

def get_labels_and_preds(test_dir, model_name, binary, sequential, image_size=256, evaluate=True):
    #counts all files in subdirectories in test folder
    NUM_IMAGES = sum(len(files) for _, _, files in os.walk(test_dir))
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    if binary:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'    
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(image_size, image_size),
        color_mode='grayscale',
        class_mode=class_mode,
        batch_size=NUM_IMAGES,
        shuffle=False)
    
    test_batch = test_gen.next()
    x = test_batch[0]
    y = test_batch[1]
    
    if binary:
        y_true=y
    else:
        y_true = np.argmax(y, axis=1)
    
    model = tf.keras.models.load_model('checkpoints/'+model_name)
    
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
                     sequential=True, image_size=256, evaluate=True, font_scale=1.2):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, binary, sequential, image_size, evaluate)
    
    y_true = rearrange_4_phase_labels(y_true)
    y_pred = rearrange_4_phase_labels(y_pred)
    
    class_names = ['Iso',
                   'N',
                   'N*', 
                   'Sm']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title,
                             font_scale=font_scale)
    
def con_mat_4_phases_2(test_dir, model_name_1, model_name_2, title='Confusion Matrix', sub_title_1='', sub_title_2='', 
                       sequential_1=True, sequential_2=True, image_size_1=256, image_size_2=256, evaluate=True, font_scale=1.2):
    y_true_1, y_pred_1 = get_labels_and_preds(test_dir, model_name_1, False, sequential_1, image_size_1, evaluate)
    
    y_true_1 = rearrange_4_phase_labels(y_true_1)
    y_pred_1 = rearrange_4_phase_labels(y_pred_1)
    
    y_true_2, y_pred_2 = get_labels_and_preds(test_dir, model_name_2, False, sequential_2, image_size_2, evaluate)
    
    y_true_2 = rearrange_4_phase_labels(y_true_2)
    y_pred_2 = rearrange_4_phase_labels(y_pred_2)
    
    class_names = ['Iso',
                   'N',
                   'N*', 
                   'Sm']
    
    display_2_confusion_matrices(y_true_1, 
                                 y_pred_1,
                                 y_true_2, 
                                 y_pred_2,
                                 class_names, 
                                 title=title,
                                 sub_title_1=sub_title_1,
                                 sub_title_2=sub_title_2,
                                 font_scale=font_scale)
    
def con_mat_smectic(test_dir, model_name, title, sequential=True, image_size=256, evaluate=True, font_scale=1.2):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, False, sequential, image_size, evaluate)
    
    class_names = ['fluid smectic',
                   'hexatic',
                   'soft crystal']
    
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

con_mat_smecticAC('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/test',
                  'smecticAC/flip_256_inception_2_3',
                  'Flip 256 inc 2',
                  font_scale=1.0)
"""
con_mat_4_phases('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test',
                 'multi train 2nd run/all_128_6',
                 title='Test set confusion matrix, 6 convolutional layers, all\n augmentations, 128 x 128 input size, 74.42% accuracy',
                 image_size=128)

con_mat_4_phases('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test',
                 'multi train 1st run/conv_2_flip_256',
                 title='Test set confusion matrix, 2 convolutional layers, flip\n augmentations, 256 x 256 input size, 94.33% accuracy',
                 image_size=256)

con_mat_4_phases_2('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test_no_iso',
                   'multi train 1st run/conv_2_flip_256',
                   'multi train 2nd run/all_128_6',
                   'Test set confusion matrices for the most and least accurate models',
                   '2 convolutional layers, flip\naugmentations, 256 x 256 input\nsize, 94.33% total accuracy',
                   '6 convolutional layers, all\naugmentations, 128 x 128 input\nsize, 74.42% total accuracy',
                   image_size_1=256,
                   image_size_2=128)

con_mat_smectic('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/test',
                'smectic/flip_256_inception_3',
                title='3 inception blocks',
                sequential=False)

con_mat_smectic('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/test',
                'smectic/flip_256_4',
                title='4 convolutional layers',
                sequential=True)
"""
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