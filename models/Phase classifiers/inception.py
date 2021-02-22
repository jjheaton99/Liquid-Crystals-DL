import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout 
from keras.layers import BatchNormalization, AveragePooling2D
from keras.layers import concatenate, Flatten, GlobalAveragePooling2D
from keras.models import Model

def inception_block(prev_output, filters=32):
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

def inception_model(num_classes, starting_channels=2, num_blocks=1, 
              double_block_channels=True, dropout_rate=0.5, input_shape=(256, 256, 1)):
    inputs = keras.Input(shape=input_shape)
    x = Conv2D(starting_channels, (7, 7), activation='relu', padding='same')(inputs)
    num_channels = starting_channels * 2
    x = BatchNormalization()(x)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D(num_channels, (1, 1), activation='relu', padding='valid')(x)
    num_channels *= 2
    x = BatchNormalization()(x)
    x = Conv2D(num_channels, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(padding='same')(x)
    
    for block in range(num_blocks):
        x = inception_block(x, num_channels)
        if double_block_channels:
            num_channels *= 2
            
    x = AveragePooling2D((5, 5), padding='valid')(x)

    x = Conv2D(num_channels, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(num_channels, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_channels // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes)(x)
    return Model(inputs=inputs, outputs=outputs)