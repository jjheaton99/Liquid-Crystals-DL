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

train_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/train'
valid_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/valid'
test_dir = 'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/test'

"""
train_gen, valid_gen, test_gen = create_generators(train_dir,
                                                   valid_dir,
                                                   test_dir,
                                                   batch_size=16)
"""

model = sequential_model(2, 3, 4)
model.summary()

"""
for i in range(1, 4):
    for j in range(1, 4):

        inc_num_blocks = i
        inc_channels = 2**(j + 1)

        train_model(
            inception_model(2, inc_num_blocks, inc_channels), 
            'inc_{0}_{1}_batch16_lr1e-4_i'.format(inc_num_blocks,
                                                    inc_channels), 
            train_gen, 
            valid_gen,
            test_gen,
            save_dir='checkpoints/ChSm/inception')
        
for i in range(1, 4):
    for j in range(1, 4):

        inc_num_blocks = i
        inc_channels = 2**(j + 1)

        train_model(
            inception_model(2, inc_num_blocks, inc_channels), 
            'inc_{0}_{1}_batch16_lr1e-4_j'.format(inc_num_blocks,
                                                    inc_channels), 
            train_gen, 
            valid_gen,
            test_gen,
            save_dir='checkpoints/ChSm/inception')
     
inc_val = np.empty((10, 9))
inc_test = np.empty((10, 9))
        
for run in range(10):
    label = chr(run+97)
    for i in range(1, 4):
        for j in range(1, 4):
            
            inc_num_blocks = i
            inc_channels = 2**(j + 1)
            
            inc_val[run][(i-1)*3+j-1], inc_test[run][(i-1)*3+j-1] = evaluate_model(
                load_model('checkpoints/ChSm/inception/inc_{0}_{1}_batch16_lr1e-4_{2}'.format(
                    inc_num_blocks,
                    inc_channels,
                    label)), 
                valid_gen, 
                test_gen)

seq_val_mean = np.mean(seq_val, axis=0)
seq_val_unc = np.std(seq_val, axis=0)

seq_val = np.round(100*np.append(seq_val, np.array([seq_val_mean, seq_val_unc]), axis=0), 2)
    
seq_test_mean = np.mean(seq_test, axis=0)
seq_test_unc = np.std(seq_test, axis=0)

seq_test = np.round(100*np.append(seq_test, np.array([seq_test_mean, seq_test_unc]), axis=0), 2)

inc_val_mean = np.mean(inc_val, axis=0)
inc_val_unc = np.std(inc_val, axis=0)

inc_val = np.round(100*np.append(inc_val, np.array([inc_val_mean, inc_val_unc]), axis=0), 2)
    
inc_test_mean = np.mean(inc_test, axis=0)
inc_test_unc = np.std(inc_test, axis=0)

inc_test = np.round(100*np.append(inc_test, np.array([inc_test_mean, inc_test_unc]), axis=0), 2)

rows = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Mean', 'Uncertainty']

seq_cols = ['3, 8', '3, 16', '3, 32', '4, 8', '4, 16', '4, 32', '5, 8', '5, 16', '5, 32']
pd.DataFrame(data=seq_val,
             index=rows,
             columns=seq_cols).to_csv('multi train results/ChACIF/seq_val_accs.csv')
pd.DataFrame(data=seq_test,
             index=rows,
             columns=seq_cols).to_csv('multi train results/ChACIF/seq_test_accs.csv')

inc_cols = ['1, 4', '1, 8', '1, 16', '2, 4', '2, 8', '2, 16', '3, 4', '3, 8', '3, 16']
pd.DataFrame(data=inc_val,
             index=rows,
             columns=inc_cols).to_csv('multi train results/ChSm/inc_val_accs.csv')
pd.DataFrame(data=inc_test,
             index=rows,
             columns=inc_cols).to_csv('multi train results/ChSm/inc_test_accs.csv')
"""