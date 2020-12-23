import numpy as np
import os

import sys
sys.path.insert(1, 'C:/MPhys project/Liquid-Crystals-DL/misc scripts')

from image_data_transformer import transform_image
from confusion_matrix_plotter import display_confusion_matrix

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

def get_labels_and_preds(test_dir, model_name, sequential, image_size=256, evaluate=True):
    #counts all files in subdirectories in test folder
    NUM_IMAGES = sum(len(files) for _, _, files in os.walk(test_dir))
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(image_size, image_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=NUM_IMAGES,
        shuffle=False)
    
    test_batch = test_gen.next()
    x = test_batch[0]
    y = test_batch[1]
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
    
    if sequential:
        y_pred = model.predict_classes(x)
        
    else:
        y_pred = np.argmax(model.predict(x), axis=1)

    return y_true, y_pred    

def con_mat_4_phases(test_dir, model_name, title='Confusion Matrix', sequential=True, image_size=256, evaluate=True):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, sequential, image_size, evaluate)
    
    #sort true and predicted labels into correct phase order 
    #to display in the confusion matrix
    def rearrange_labels(labels):
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
    
    y_true = rearrange_labels(y_true)
    y_pred = rearrange_labels(y_pred)
    
    class_names = ['isotropic',
                   'nematic',
                   'cholesteric', 
                   'smectic']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title)
    
def con_mat_smectic(test_dir, model_name, title, sequential=True, image_size=256, evaluate=True):
    y_true, y_pred = get_labels_and_preds(test_dir, model_name, sequential, image_size, evaluate)
    
    class_names = ['fluid smectic',
                   'hexatic',
                   'soft crystal']
    
    display_confusion_matrix(y_true, 
                             y_pred, 
                             class_names, 
                             title=title)

con_mat_4_phases('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/set2/test',
                 'multi train 1st run/conv_2_flip_256',
                 title='Test set confusion matrix, 2 convolutional layers,\n flip augmentations, 256 x 256 input size')

"""    
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