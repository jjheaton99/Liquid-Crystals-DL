import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Flatten

def sequential_model(num_classes, num_layers, starting_channels, channel_doubling=True, 
               dropout_rate=0.5, input_shape=(256, 256, 1)):
    x = inputs = keras.Input(shape=input_shape)
    num_channels = starting_channels
        
    for layer in range(num_layers - 1):
        x = Conv2D(num_channels, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
        if channel_doubling:
            num_channels *= 2
        
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