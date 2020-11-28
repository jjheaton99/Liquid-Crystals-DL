#Sequential 4 phases model v3 1 layer

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

INPUT_SHAPE_256 = (256, 256, 1)
INPUT_SHAPE_128 = (128, 128, 1)
NUM_CLASSES = 4

all_256_1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_256_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_256_3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(), 
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_256_4 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_256_5 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_256_6 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(1024, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])
##############################################################################

flip_256_1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_256_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_256_3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(), 
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_256_4 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_256_5 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_256_6 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_256),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(1024, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])
##############################################################################

all_128_1 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_128_2 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_128_3 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(), 
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_128_4 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_128_5 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

all_128_6 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])
##############################################################################

flip_128_1 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_128_2 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_128_3 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(), 
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),      
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_128_4 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_128_5 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])

flip_128_6 = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_128),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),   
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),       
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES)
    ])