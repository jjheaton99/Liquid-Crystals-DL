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

train_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4-phase/train'
valid_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4-phase/valid'
test_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/4-phase/test'

train_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/train'
valid_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/valid'
test_dir_smectic = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic/test'

train_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/train'
valid_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/valid'
test_dir_smecticAC = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smectic A C/test'

train_gen, valid_gen, test_gen = create_generators(train_dir_smecticAC,
                                                   valid_dir_smecticAC,
                                                   test_dir_smecticAC)
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
                          num_classes=2,
                          patch_dim=16,
                          model_dim=64,
                          num_encoders=4,
                          dropout_rate=0.0)

print(train_model(vit, 
            'smecticAC_1', 
            train_gen, 
            valid_gen,
            test_gen,
            save_dir='checkpoints/vis_trans',
            is_vit=True))

"""
inc_val_accs = np.empty((3, 3))
inc_test_accs = np.empty((3, 3))

inc_val_accs[0][0], inc_test_accs[0][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_1_1'),
                    valid_gen,
                    test_gen)

inc_val_accs[0][1], inc_test_accs[0][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_2_1'),
                    valid_gen,
                    test_gen)

inc_val_accs[0][2], inc_test_accs[0][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_3_1'),
                    valid_gen,
                    test_gen)

inc_val_accs[1][0], inc_test_accs[1][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_1_2'),
                    valid_gen,
                    test_gen)

inc_val_accs[1][1], inc_test_accs[1][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_2_2'),
                    valid_gen,
                    test_gen)

inc_val_accs[1][2], inc_test_accs[1][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_3_2'),
                    valid_gen,
                    test_gen)

inc_val_accs[2][0], inc_test_accs[2][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_1_3'),
                    valid_gen,
                    test_gen)

inc_val_accs[2][1], inc_test_accs[2][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_2_3'),
                    valid_gen,
                    test_gen)

inc_val_accs[2][2], inc_test_accs[2][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_inception_3_3'),
                    valid_gen,
                    test_gen)

seq_val_accs = np.empty((3, 6))
seq_test_accs = np.empty((3, 6))

seq_val_accs[0][0], seq_test_accs[0][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_1_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[0][1], seq_test_accs[0][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_2_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[0][2], seq_test_accs[0][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_3_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[0][3], seq_test_accs[0][3] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_4_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[0][4], seq_test_accs[0][4] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_5_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[0][5], seq_test_accs[0][5] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_6_1'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][0], seq_test_accs[1][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_1_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][1], seq_test_accs[1][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_2_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][2], seq_test_accs[1][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_3_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][3], seq_test_accs[1][3] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_4_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][4], seq_test_accs[1][4] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_5_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[1][5], seq_test_accs[1][5] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_6_2'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][0], seq_test_accs[2][0] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_1_3'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][1], seq_test_accs[2][1] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_2_3'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][2], seq_test_accs[2][2] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_3_3'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][3], seq_test_accs[2][3] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_4_3'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][4], seq_test_accs[2][4] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_5_3'),
                    valid_gen,
                    test_gen)

seq_val_accs[2][5], seq_test_accs[2][5] = evaluate_model(
                    load_model('checkpoints/smecticAC/flip_256_6_3'),
                    valid_gen,
                    test_gen)

inc_val_mean = np.mean(inc_val_accs, axis=0)
inc_val_unc = np.empty(3)
for index in range(3):
    transpose_accs_idx = np.transpose(inc_val_accs)[index]
    inc_val_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2
inc_val_accs = np.round(100*np.append(inc_val_accs, np.array([inc_val_mean, inc_val_unc]), axis=0), 2)
    
inc_test_mean = np.mean(inc_test_accs, axis=0)
inc_test_unc = np.empty(3)
for index in range(3):
    transpose_accs_idx = np.transpose(inc_test_accs)[index]
    inc_test_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2
inc_test_accs = np.round(100*np.append(inc_test_accs, np.array([inc_test_mean, inc_test_unc]), axis=0), 2)

seq_val_mean = np.mean(seq_val_accs, axis=0)
seq_val_unc = np.empty(6)
for index in range(6):
    transpose_accs_idx = np.transpose(seq_val_accs)[index]
    seq_val_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2
seq_val_accs = np.round(100*np.append(seq_val_accs, np.array([seq_val_mean, seq_val_unc]), axis=0), 2)

seq_test_mean = np.mean(seq_test_accs, axis=0)
seq_test_unc = np.empty(6)
for index in range(6):
    transpose_accs_idx = np.transpose(seq_test_accs)[index]
    seq_test_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2
seq_test_accs = np.round(100*np.append(seq_test_accs, np.array([seq_test_mean, seq_test_unc]), axis=0), 2)

rows = ['1st run', '2nd run', '3rd run', 'Mean', 'Uncertainty']
inc_cols = ['1', '2', '3']
seq_cols = ['1', '2', '3', '4', '5', '6']

pd.DataFrame(data=inc_val_accs,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smecticAC/inc_val_accs.csv')

pd.DataFrame(data=inc_test_accs,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smecticAC/inc_test_accs.csv')

pd.DataFrame(data=seq_val_accs,
             index=rows,
             columns=seq_cols).to_csv('multi train results/smecticAC/seq_val_accs.csv')

pd.DataFrame(data=seq_test_accs,
             index=rows,
             columns=seq_cols).to_csv('multi train results/smecticAC/seq_test_accs.csv')
"""