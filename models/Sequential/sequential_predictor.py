import numpy as np

import sys
sys.path.insert(1, 'D:/MPhys project/Liquid-Crystals-DL/misc scripts')

from image_data_transformer import transform_image

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_gen = test_datagen.flow_from_directory(
    directory='D:/MPhys project/Liquid-Crystals-DL/data/Prepared data/test',
    target_size=(200, 200),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    shuffle=False)
TEST_STEP_SIZE = test_gen.n//test_gen.batch_size

model = tf.keras.models.load_model('checkpoints', compile=True)

model.evaluate(
    test_gen,
    steps=TEST_STEP_SIZE,
    verbose=2)

#outputs prediction for image file and associated confidence
def predict_image(filename, model=model):
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
        print('columnar')
    elif pred_class[0] == 2:
        print('isotropic')
    elif pred_class[0] == 3:
        print('nematic')
    elif pred_class[0] == 4:
        print('smectic')
    elif pred_class[0] == 5:
        print('twist_grain_boundary')
        
    print('Confidence level:')
    print(probs[0][pred_class[0]])
        
    return (pred_class[0], probs[0][pred_class[0]])

predict_image('test_image1.jpg')
predict_image('test_image2.jpg')
predict_image('test_image3.jpg')