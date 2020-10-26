import numpy as np
import os

import sys
sys.path.insert(1, 'D:/MPhys project/Liquid-Crystals-DL/misc scripts')

from image_data_transformer import transform_image
from confusion_matrix_plotter import display_confusion_matrix

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

test_dir = 'D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/test'
#counts all files in subdirectories in test folder
BATCH_SIZE = sum(len(files) for _, _, files in os.walk(test_dir))

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_gen = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(200, 200),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False)

test_batch = test_gen.next()
y_true = np.argmax(test_batch[1], axis=1)

model = tf.keras.models.load_model('checkpoints/v2', compile=True)

#evaluate for total test set accuracy
model.evaluate(
    test_gen,
    steps=1,
    verbose=2)

y_pred = model.predict_classes(test_batch[0])

class_names = ['cholesteric', 
               'isotropic', 
               'nematic', 
               'smectic']

display_confusion_matrix(y_true, y_pred, class_names)

#outputs prediction for image file and associated confidence
def predict_image(filename, model=model, show=False):
    if show:
        transform_image(Image.open(filename)).show()
    image = transform_image(Image.open(filename), as_array=True)/255.0
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

predict_image('test_image_nematic.jpg')
predict_image('test_image_cholesteric.jpg')
predict_image('test_image_smectic.jpg')
