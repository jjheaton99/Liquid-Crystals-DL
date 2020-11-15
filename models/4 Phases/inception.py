import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout 
from keras.layers import BatchNormalization, AveragePooling2D
from keras.layers import concatenate, Flatten
from keras.models import Model

def convBlock(prev_output, filters=32, kernel_size=(3, 3), num_layers=1, 
              max_pooling=False, pool_size=(3, 3), padding='same'):
    block = Conv2D(filters=filters, 
                   kernel_size=kernel_size, 
                   padding=padding, 
                   activation='relu')(prev_output)
    block = BatchNormalization()(block)
    
    if num_layers < 2 and max_pooling:
        block = MaxPooling2D(pool_size=pool_size, padding='same')(block)
    else:
        for i in range(num_layers-1):
            block = Conv2D(filters=filters, 
                           kernel_size=kernel_size, 
                           padding=padding, 
                           activation='relu')(block)
            block = BatchNormalization()(block)
        if max_pooling:
            block = MaxPooling2D(pool_size=pool_size, padding='same')(block)
    return block

def denseBlock(prev_output, units=10, dropout_rate=0.5, num_layers=1):
    block = Dense(units=units, activation='relu')(prev_output)
    block = BatchNormalization()(block)
    block = Dropout(rate=dropout_rate)(block)
    if num_layers > 1:
        for i in range(num_layers-1):
                block = Dense(units=units, activation='relu')(block)
                block = BatchNormalization()(block)
                block = Dropout(rate=dropout_rate)(block)
    return block

def inceptionBlock(prev_output, filters=32):
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

INPUT_SHAPE = (256, 256, 1)
NUM_CLASSES = 4

img_input = keras.Input(shape=INPUT_SHAPE)

conv_block_1 = convBlock(img_input, 
                         filters=16, 
                         kernel_size=(7, 7), 
                         num_layers=1, 
                         max_pooling=True)
conv_block_2 = convBlock(conv_block_1, 
                         filters=32, 
                         kernel_size=(1, 1), 
                         num_layers=1)
conv_block_3 = convBlock(conv_block_2, 
                         filters=64, 
                         num_layers=2, 
                         max_pooling=True)

inception_block_1 = inceptionBlock(conv_block_3, filters=128)
inception_block_2 = inceptionBlock(inception_block_1, filters=256)
inception_block_3 = inceptionBlock(inception_block_2, filters=512)

average_pool = AveragePooling2D(pool_size=(5, 5))(inception_block_3)

conv_block_4 = convBlock(average_pool, 
                         filters=512, 
                         kernel_size=(3, 3), 
                         num_layers=1)

flatten = Flatten()(conv_block_4)
dense_block1 = denseBlock(flatten, 
                          units=128, 
                          dropout_rate=0.5, 
                          num_layers=2)
output = Dense(NUM_CLASSES)(dense_block1)

model = Model(inputs=img_input, outputs=output)