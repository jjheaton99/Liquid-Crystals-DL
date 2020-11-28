import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout 
from keras.layers import BatchNormalization, AveragePooling2D
from keras.layers import concatenate, Flatten
from keras.models import Model

def block(prev_output, filters=32):
    block_1x1 = Conv2D(filters, 
                       (1, 1), 
                       padding='same', 
                       activation='relu')(prev_output)
    block_1x1 = BatchNormalization()(block_1x1)
    
    block_3x3 = Conv2D(filters, 
                       (1, 1), 
                       padding='same', 
                       activation='relu')(prev_output)
    block_3x3 = Conv2D(filters, 
                       (3, 3), 
                       padding='same', 
                       activation='relu')(block_3x3)
    block_3x3 = BatchNormalization()(block_3x3)
    
    block_5x5 = Conv2D(filters, 
                       (1, 1), 
                       padding='same', 
                       activation='relu')(prev_output)
    block_5x5 = Conv2D(filters, 
                       (5, 5), 
                       padding='same', 
                       activation='relu')(block_5x5)
    block_5x5 = BatchNormalization()(block_5x5)
    
    block_pooling = MaxPooling2D(pool_size=(3, 3), 
                                 strides=(1, 1), 
                                 padding='same')(prev_output)
    block_pooling = Conv2D(filters, 
                           (1, 1), 
                           padding='same', 
                           activation='relu')(block_pooling)
    block_pooling = BatchNormalization()(block_pooling)
    
    return concatenate([block_1x1, block_3x3, block_5x5, block_pooling], axis=3)