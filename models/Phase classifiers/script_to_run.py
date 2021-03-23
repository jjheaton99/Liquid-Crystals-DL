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

train_dir_smecticIF = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticIF/train'
valid_dir_smecticIF = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticIF/valid'
test_dir_smecticIF = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/smecticIF/test'

train_dir_ch_sm3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm3/train'
valid_dir_ch_sm3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm3/valid'
test_dir_ch_sm3 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm3/test'

train_dir_ch_sm4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm4/train'
valid_dir_ch_sm4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm4/valid'
test_dir_ch_sm4 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm4/test'

train_dir_ch_sm5 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm5/train'
valid_dir_ch_sm5 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm5/valid'
test_dir_ch_sm5 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm5/test'

train_dir_ch_sm6 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm6/train'
valid_dir_ch_sm6 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm6/valid'
test_dir_ch_sm6 = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ch_sm6/test'

train_gen, valid_gen, test_gen = create_generators(train_dir_smecticIF,
                                                   valid_dir_smecticIF,
                                                   test_dir_smecticIF,
                                                   batch_size=16)
"""
train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(sequential_model(2, 3, 8),
            'seq_3_8_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 16),
            'seq_3_16_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 3, 32),
            'seq_3_32_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 8),
            'seq_4_8_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 16),
            'seq_4_16_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')
train_model(sequential_model(2, 4, 32),
            'seq_4_32_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/sequential')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_a',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_b',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_c',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_d',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_e',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_f',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_g',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_h',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_i',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

train_model(inception_model(2, 1, 2),
            'inc_1_2_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 4),
            'inc_1_4_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 1, 8),
            'inc_1_8_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 2),
            'inc_2_2_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 4),
            'inc_2_4_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 2, 8),
            'inc_2_8_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 2),
            'inc_3_2_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 4),
            'inc_3_4_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')
train_model(inception_model(2, 3, 8),
            'inc_3_8_batch16_lr1e-4_j',
            train_gen,
            valid_gen,
            test_gen,
            save_dir='checkpoints/smecticIF/inception')

"""
seq_val = np.empty((10, 6))
seq_test = np.empty((10, 6))

seq_val[0][0], seq_test[0][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][0], seq_test[1][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][0], seq_test[2][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][0], seq_test[3][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][0], seq_test[4][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][0], seq_test[5][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][0], seq_test[6][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][0], seq_test[7][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][0], seq_test[8][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][0], seq_test[9][0] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_8_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val[0][1], seq_test[0][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][1], seq_test[1][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][1], seq_test[2][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][1], seq_test[3][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][1], seq_test[4][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][1], seq_test[5][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][1], seq_test[6][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][1], seq_test[7][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][1], seq_test[8][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][1], seq_test[9][1] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_16_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val[0][2], seq_test[0][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][2], seq_test[1][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][2], seq_test[2][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][2], seq_test[3][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][2], seq_test[4][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][2], seq_test[5][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][2], seq_test[6][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][2], seq_test[7][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][2], seq_test[8][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][2], seq_test[9][2] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_3_32_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val[0][3], seq_test[0][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][3], seq_test[1][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][3], seq_test[2][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][3], seq_test[3][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][3], seq_test[4][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][3], seq_test[5][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][3], seq_test[6][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][3], seq_test[7][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][3], seq_test[8][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][3], seq_test[9][3] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_8_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val[0][4], seq_test[0][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][4], seq_test[1][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][4], seq_test[2][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][4], seq_test[3][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][4], seq_test[4][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][4], seq_test[5][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][4], seq_test[6][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][4], seq_test[7][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][4], seq_test[8][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][4], seq_test[9][4] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_16_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val[0][5], seq_test[0][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
seq_val[1][5], seq_test[1][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
seq_val[2][5], seq_test[2][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
seq_val[3][5], seq_test[3][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
seq_val[4][5], seq_test[4][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
seq_val[5][5], seq_test[5][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
seq_val[6][5], seq_test[6][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
seq_val[7][5], seq_test[7][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
seq_val[8][5], seq_test[8][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
seq_val[9][5], seq_test[9][5] = evaluate_model(
    load_model('checkpoints/smecticIF/sequential/seq_4_32_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

seq_val_mean = np.mean(seq_val, axis=0)
seq_val_unc = np.std(seq_val, axis=0)

seq_val = np.round(100*np.append(seq_val, np.array([seq_val_mean, seq_val_unc]), axis=0), 2)
    
seq_test_mean = np.mean(seq_test, axis=0)
seq_test_unc = np.std(seq_test, axis=0)

seq_test = np.round(100*np.append(seq_test, np.array([seq_test_mean, seq_test_unc]), axis=0), 2)

rows = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Mean', 'Uncertainty']

seq_cols = ['3, 8', '3, 16', '3, 32', '4, 8', '4, 16', '4, 32']
pd.DataFrame(data=seq_val,
             index=rows,
             columns=seq_cols).to_csv('multi train results/smecticIF/seq_val_accs.csv')
pd.DataFrame(data=seq_test,
             index=rows,
             columns=seq_cols).to_csv('multi train results/smecticIF/seq_test_accs.csv')

inc_val = np.empty((10, 9))
inc_test = np.empty((10, 9))

inc_val[0][0], inc_test[0][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][0], inc_test[1][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][0], inc_test[2][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][0], inc_test[3][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][0], inc_test[4][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][0], inc_test[5][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][0], inc_test[6][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][0], inc_test[7][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][0], inc_test[8][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][0], inc_test[9][0] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_2_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][1], inc_test[0][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][1], inc_test[1][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][1], inc_test[2][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][1], inc_test[3][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][1], inc_test[4][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][1], inc_test[5][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][1], inc_test[6][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][1], inc_test[7][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][1], inc_test[8][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][1], inc_test[9][1] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_4_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][2], inc_test[0][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][2], inc_test[1][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][2], inc_test[2][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][2], inc_test[3][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][2], inc_test[4][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][2], inc_test[5][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][2], inc_test[6][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][2], inc_test[7][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][2], inc_test[8][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][2], inc_test[9][2] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_1_8_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][3], inc_test[0][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][3], inc_test[1][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][3], inc_test[2][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][3], inc_test[3][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][3], inc_test[4][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][3], inc_test[5][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][3], inc_test[6][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][3], inc_test[7][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][3], inc_test[8][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][3], inc_test[9][3] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_2_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][4], inc_test[0][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][4], inc_test[1][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][4], inc_test[2][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][4], inc_test[3][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][4], inc_test[4][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][4], inc_test[5][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][4], inc_test[6][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][4], inc_test[7][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][4], inc_test[8][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][4], inc_test[9][4] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_4_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][5], inc_test[0][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][5], inc_test[1][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][5], inc_test[2][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][5], inc_test[3][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][5], inc_test[4][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][5], inc_test[5][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][5], inc_test[6][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][5], inc_test[7][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][5], inc_test[8][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][5], inc_test[9][5] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_2_8_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][6], inc_test[0][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][6], inc_test[1][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][6], inc_test[2][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][6], inc_test[3][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][6], inc_test[4][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][6], inc_test[5][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][6], inc_test[6][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][6], inc_test[7][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][6], inc_test[8][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][6], inc_test[9][6] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_2_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][7], inc_test[0][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][7], inc_test[1][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][7], inc_test[2][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][7], inc_test[3][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][7], inc_test[4][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][7], inc_test[5][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][7], inc_test[6][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][7], inc_test[7][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][7], inc_test[8][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][7], inc_test[9][7] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_4_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val[0][8], inc_test[0][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_a'), 
    valid_gen, 
    test_gen)
inc_val[1][8], inc_test[1][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_b'), 
    valid_gen, 
    test_gen)
inc_val[2][8], inc_test[2][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_c'), 
    valid_gen, 
    test_gen)
inc_val[3][8], inc_test[3][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_d'), 
    valid_gen, 
    test_gen)
inc_val[4][8], inc_test[4][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_e'), 
    valid_gen, 
    test_gen)
inc_val[5][8], inc_test[5][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_f'), 
    valid_gen, 
    test_gen)
inc_val[6][8], inc_test[6][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_g'), 
    valid_gen, 
    test_gen)
inc_val[7][8], inc_test[7][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_h'), 
    valid_gen, 
    test_gen)
inc_val[8][8], inc_test[8][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_i'), 
    valid_gen, 
    test_gen)
inc_val[9][8], inc_test[9][8] = evaluate_model(
    load_model('checkpoints/smecticIF/inception/inc_3_8_batch16_lr1e-4_j'), 
    valid_gen, 
    test_gen)

inc_val_mean = np.mean(inc_val, axis=0)
inc_val_unc = np.std(inc_val, axis=0)

inc_val = np.round(100*np.append(inc_val, np.array([inc_val_mean, inc_val_unc]), axis=0), 2)
    
inc_test_mean = np.mean(inc_test, axis=0)
inc_test_unc = np.std(inc_test, axis=0)

inc_test = np.round(100*np.append(inc_test, np.array([inc_test_mean, inc_test_unc]), axis=0), 2)

inc_cols = ['1, 2', '1, 4', '1, 8', '2, 2', '2, 4', '2, 8', '3, 2', '3, 4', '3, 8']
pd.DataFrame(data=inc_val,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smecticIF/inc_val_accs.csv')
pd.DataFrame(data=inc_test,
             index=rows,
             columns=inc_cols).to_csv('multi train results/smecticIF/inc_test_accs.csv')
