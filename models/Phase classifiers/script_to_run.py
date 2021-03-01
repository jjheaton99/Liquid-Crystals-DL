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
train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2a',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2b',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2c',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2d',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2e',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2f',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2g',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2h',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2i',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2j',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2k',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

train_model(inception_model(num_classes=3, num_blocks=2),
            'fl_inc_2l',
            train_gen, 
            valid_gen, 
            save_dir='checkpoints/smectic3/sem2',
            loss='fl')

"""
inc_val_accs = np.empty((12, 3))
inc_test_accs = np.empty((12, 3))

inc_val_accs[0][0], inc_test_accs[0][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1a'),
                    valid_gen,
                    test_gen)
inc_val_accs[1][0], inc_test_accs[1][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1b'),
                    valid_gen,
                    test_gen)
inc_val_accs[2][0], inc_test_accs[2][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1c'),
                    valid_gen,
                    test_gen)
inc_val_accs[3][0], inc_test_accs[3][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1d'),
                    valid_gen,
                    test_gen)
inc_val_accs[4][0], inc_test_accs[4][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1e'),
                    valid_gen,
                    test_gen)
inc_val_accs[5][0], inc_test_accs[5][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1f'),
                    valid_gen,
                    test_gen)
inc_val_accs[6][0], inc_test_accs[6][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1g'),
                    valid_gen,
                    test_gen)
inc_val_accs[7][0], inc_test_accs[7][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1h'),
                    valid_gen,
                    test_gen)
inc_val_accs[8][0], inc_test_accs[8][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1i'),
                    valid_gen,
                    test_gen)
inc_val_accs[9][0], inc_test_accs[9][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1j'),
                    valid_gen,
                    test_gen)
inc_val_accs[10][0], inc_test_accs[10][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1k'),
                    valid_gen,
                    test_gen)
inc_val_accs[11][0], inc_test_accs[11][0] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_1l'),
                    valid_gen,
                    test_gen)

inc_val_accs[0][1], inc_test_accs[0][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2a'),
                    valid_gen,
                    test_gen)
inc_val_accs[1][1], inc_test_accs[1][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2b'),
                    valid_gen,
                    test_gen)
inc_val_accs[2][1], inc_test_accs[2][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2c'),
                    valid_gen,
                    test_gen)
inc_val_accs[3][1], inc_test_accs[3][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2d'),
                    valid_gen,
                    test_gen)
inc_val_accs[4][1], inc_test_accs[4][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2e'),
                    valid_gen,
                    test_gen)
inc_val_accs[5][1], inc_test_accs[5][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2f'),
                    valid_gen,
                    test_gen)
inc_val_accs[6][1], inc_test_accs[6][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2g'),
                    valid_gen,
                    test_gen)
inc_val_accs[7][1], inc_test_accs[7][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2h'),
                    valid_gen,
                    test_gen)
inc_val_accs[8][1], inc_test_accs[8][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2i'),
                    valid_gen,
                    test_gen)
inc_val_accs[9][1], inc_test_accs[9][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2j'),
                    valid_gen,
                    test_gen)
inc_val_accs[10][1], inc_test_accs[10][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2k'),
                    valid_gen,
                    test_gen)
inc_val_accs[11][1], inc_test_accs[11][1] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_2l'),
                    valid_gen,
                    test_gen)

inc_val_accs[0][2], inc_test_accs[0][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3a'),
                    valid_gen,
                    test_gen)
inc_val_accs[1][2], inc_test_accs[1][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3b'),
                    valid_gen,
                    test_gen)
inc_val_accs[2][2], inc_test_accs[2][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3c'),
                    valid_gen,
                    test_gen)
inc_val_accs[3][2], inc_test_accs[3][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3d'),
                    valid_gen,
                    test_gen)
inc_val_accs[4][2], inc_test_accs[4][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3e'),
                    valid_gen,
                    test_gen)
inc_val_accs[5][2], inc_test_accs[5][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3f'),
                    valid_gen,
                    test_gen)
inc_val_accs[6][2], inc_test_accs[6][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3g'),
                    valid_gen,
                    test_gen)
inc_val_accs[7][2], inc_test_accs[7][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3h'),
                    valid_gen,
                    test_gen)
inc_val_accs[8][2], inc_test_accs[8][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3i'),
                    valid_gen,
                    test_gen)
inc_val_accs[9][2], inc_test_accs[9][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3j'),
                    valid_gen,
                    test_gen)
inc_val_accs[10][2], inc_test_accs[10][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3k'),
                    valid_gen,
                    test_gen)
inc_val_accs[11][2], inc_test_accs[11][2] = evaluate_model(
                    load_model('checkpoints/smectic3/sem2/inc_3l'),
                    valid_gen,
                    test_gen)

inc_val_mean = np.mean(inc_val_accs, axis=0)
inc_val_unc = np.std(inc_val_accs, axis=0)

for index in range(3):
    transpose_accs_idx = np.transpose(inc_val_accs)[index]
    inc_val_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2

inc_val_accs = np.round(100*np.append(inc_val_accs, np.array([inc_val_mean, inc_val_unc]), axis=0), 2)
    
inc_test_mean = np.mean(inc_test_accs, axis=0)
inc_test_unc = np.std(inc_test_accs, axis=0)

for index in range(3):
    transpose_accs_idx = np.transpose(inc_test_accs)[index]
    inc_test_unc[index] = (np.max(transpose_accs_idx) - np.min(transpose_accs_idx)) / 2

inc_test_accs = np.round(100*np.append(inc_test_accs, np.array([inc_test_mean, inc_test_unc]), axis=0), 2)

rows = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Mean', 'Uncertainty']
inc_cols = ['1', '2', '3']
pd.DataFrame(data=inc_val_accs,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smectic3/inc_val_accs.csv')
pd.DataFrame(data=inc_test_accs,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smectic3/inc_test_accs.csv')
