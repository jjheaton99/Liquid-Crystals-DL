#Smectic phases A, C models

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Flatten

import inception

INPUT_SHAPE_256 = (256, 256, 1)

img_input = keras.Input(shape=INPUT_SHAPE_256)

#1 inception block
##############################################################################
conv_1 = Conv2D(2, (7, 7), activation='relu', padding='same')(img_input)
batch_norm_1 = BatchNormalization()(conv_1)
max_pool_1 = MaxPooling2D(padding='same')(batch_norm_1)
conv_2 = Conv2D(4, (1, 1), activation='relu', padding='valid')(max_pool_1)
batch_norm_2 = BatchNormalization()(conv_2)
conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(batch_norm_2)
batch_norm_3 = BatchNormalization()(conv_3)
max_pool_2 = MaxPooling2D(padding='same')(batch_norm_3)

inception_1 = inception.block(max_pool_2, 8)

average_pool = AveragePooling2D((5, 5), padding='valid')(inception_1)

conv_4 = Conv2D(16, (3, 3), activation='relu', padding='same')(average_pool)
batch_norm_3 = BatchNormalization()(conv_4)

global_average = GlobalAveragePooling2D()(batch_norm_3)

dense_1 = Dense(16, activation='relu')(global_average)
batch_norm_4 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.5)(batch_norm_4)
dense_2 = Dense(8, activation='relu')(dropout_1)
batch_norm_5 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.5)(batch_norm_5)

output = Dense(1)(dropout_2)

flip_256_inception_1 = keras.models.Model(inputs=img_input, outputs=output)
##############################################################################

#2 inception blocks
##############################################################################
conv_1 = Conv2D(2, (7, 7), activation='relu', padding='same')(img_input)
batch_norm_1 = BatchNormalization()(conv_1)
max_pool_1 = MaxPooling2D(padding='same')(batch_norm_1)
conv_2 = Conv2D(4, (1, 1), activation='relu', padding='valid')(max_pool_1)
batch_norm_2 = BatchNormalization()(conv_2)
conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(batch_norm_2)
batch_norm_3 = BatchNormalization()(conv_3)
max_pool_2 = MaxPooling2D(padding='same')(batch_norm_3)

inception_1 = inception.block(max_pool_2, 8)
inception_2 = inception.block(inception_1, 16)

average_pool = AveragePooling2D((5, 5), padding='valid')(inception_2)

conv_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(average_pool)
batch_norm_3 = BatchNormalization()(conv_4)

global_average = GlobalAveragePooling2D()(batch_norm_3)

dense_1 = Dense(32, activation='relu')(global_average)
batch_norm_4 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.5)(batch_norm_4)
dense_2 = Dense(16, activation='relu')(dropout_1)
batch_norm_5 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.5)(batch_norm_5)

output = Dense(1)(dropout_2)

flip_256_inception_2 = keras.models.Model(inputs=img_input, outputs=output)
##############################################################################

#3 inception blocks
##############################################################################
conv_1 = Conv2D(2, (7, 7), activation='relu', padding='same')(img_input)
batch_norm_1 = BatchNormalization()(conv_1)
max_pool_1 = MaxPooling2D(padding='same')(batch_norm_1)
conv_2 = Conv2D(4, (1, 1), activation='relu', padding='valid')(max_pool_1)
batch_norm_2 = BatchNormalization()(conv_2)
conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(batch_norm_2)
batch_norm_3 = BatchNormalization()(conv_3)
max_pool_2 = MaxPooling2D(padding='same')(batch_norm_3)

inception_1 = inception.block(max_pool_2, 8)
inception_2 = inception.block(inception_1, 16)
inception_3 = inception.block(inception_2, 32)

average_pool = AveragePooling2D((5, 5), padding='valid')(inception_3)

conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(average_pool)
batch_norm_3 = BatchNormalization()(conv_4)

global_average = GlobalAveragePooling2D()(batch_norm_3)

dense_1 = Dense(64, activation='relu')(global_average)
batch_norm_4 = BatchNormalization()(dense_1)
dropout_1 = Dropout(0.5)(batch_norm_4)
dense_2 = Dense(32, activation='relu')(dropout_1)
batch_norm_5 = BatchNormalization()(dense_2)
dropout_2 = Dropout(0.5)(batch_norm_5)

output = Dense(1)(dropout_2)

flip_256_inception_3 = keras.models.Model(inputs=img_input, outputs=output)
##############################################################################
