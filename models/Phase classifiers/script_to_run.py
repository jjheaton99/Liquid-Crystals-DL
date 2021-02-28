import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.models import load_model, save_model
from kerastuner.tuners import Hyperband
from keras.utils.vis_utils import plot_model

from model_training import create_generators, train_model, evaluate_model, load_model_fl

from sequential import sequential_model
from inception import inception_model
from vision_transformer import vision_transformer_model

import ViT_hypermodel

train_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/train'
valid_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/valid'
test_dir_4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4phases/test'

train_dir_smectic3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/train'
valid_dir_smectic3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/valid'
test_dir_smectic3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic3/test'

train_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/train'
valid_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/valid'
test_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticAC/test'

train_gen, valid_gen, test_gen = create_generators(train_dir_smectic3,
                                                   valid_dir_smectic3,
                                                   test_dir_smectic3,
                                                   batch_size=64)
"""
train_model(inception_model(num_classes=3, num_blocks=3),
            'fl_inc_3a',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2',
            loss='fl',
            patience=50)
"""
model = load_model_fl('checkpoints/smectic3/sem2/fl_inc_3a')
evaluate_model(model, valid_gen, test_gen)
"""
train_model(inception_model(num_classes=3, num_blocks=3),
            
            'inc_3j',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=3),
            'inc_3k',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=3),
            'inc_3l',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=2),
            'inc_2j',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=2),
            'inc_2k',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=2),
            'inc_2l',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=1),
            'inc_1j',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=1),
            'inc_1k',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')

train_model(inception_model(num_classes=3, num_blocks=1),
            'inc_1l',
            train_gen, 
            valid_gen, 
            test_gen,
            save_dir='checkpoints/smectic3/sem2')
"""

"""
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model_save = tf.keras.callbacks.ModelCheckpoint('checkpoints/vis_trans', save_best_only=True)
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=20,
                                           verbose=1,
                                           min_lr=1e-5)

hypermodel = ViT_hypermodel.ViTHypermodel((256, 256, 1), 3)

tuner = Hyperband(hypermodel,
                  objective='val_accuracy',
                  max_epochs=100,
                  directory='hyperband',
                  project_name='ViT_smectic3')

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

save_model(best_model, 'checkpoints/ViT/best_hp_model_smectic3')
"""
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
            save_dir='checkpoints/ViT',
            is_vit=True))
"""