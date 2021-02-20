import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.models import load_model, save_model
from kerastuner.tuners import Hyperband

from model_training import create_generators, train_model, evaluate_model

import v1_4_phases
import v2_4_phases
import v3_4_phases
import smectic_models
import smecticAC_models
import vision_transformer
import ViT_hypermodel

train_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/train'
valid_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/valid'
test_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/test'

train_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/train'
valid_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/valid'
test_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/test'

train_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/train'
valid_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/valid'
test_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/test'

train_gen, valid_gen, test_gen = create_generators(train_dir_smectic,
                                                   valid_dir_smectic,
                                                   test_dir_smectic)
"""
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model_save = tf.keras.callbacks.ModelCheckpoint('checkpoints/vis_trans', save_best_only=True)
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=20,
                                           verbose=1,
                                           min_lr=1e-5)

hypermodel = vis_trans_hypermodel.VisTransHypermodel((256, 256, 1), 4)

tuner = Hyperband(hypermodel,
                  objective='val_accuracy',
                  max_epochs=100,
                  directory=os.path.normpath('C:/MPhys project/Liquid-Crystals-DL/models/Phase classifiers/hyperband'),
                  project_name='ViT')

tuner.search(x=train_gen,
             steps_per_epoch=train_gen.n//train_gen.batch_size,
             epochs=100,
             callbacks=[early_stop, learning_rate_schedule],
             validation_data=valid_gen,
             validation_steps=valid_gen.n//valid_gen.batch_size)

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters()[0]

print('Optimal hyperparameters:')
print('lr: ', best_hps.get('learning_rate'))
print('patch_dim: ', best_hps.get('patch_dim'))
print('model_dim: ', best_hps.get('model_dim'))
print('num_encoders: ', best_hps.get('num_encoders'))
print('dropout rate: ', best_hps.get('dropout_rate'))

best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy = best_model.evaluate(test_gen,
                    steps=test_gen.n//test_gen.batch_size,
                    verbose=2)

save_model(best_model, 'checkpoints/ViT/best_hp_model')
"""

vit = vision_transformer.VisionTransformer(input_shape=(256, 256, 1),
                          num_classes=3,
                          patch_dim=16,
                          model_dim=128,
                          num_encoders=8,
                          dropout_rate=0.1)

print(train_model(vit, 
            'smectic3_2', 
            train_gen, 
            valid_gen,
            test_gen,
            save_dir='checkpoints/vis_trans',
            is_vit=True))