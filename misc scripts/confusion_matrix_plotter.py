import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix

def display_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', font_scale=1.0):
    con_mat = confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=class_names, columns=class_names)
    
    figsize=(len(class_names), len(class_names))
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True)
    plt.tight_layout()
    plt.ylabel('True phase')
    plt.xlabel('Predicted phase')
    plt.show()
    
def display_2_confusion_matrices(y_true_1, y_pred_1, y_true_2, y_pred_2, class_names, 
                                 title='Confusion Matrix', sub_title_1='', sub_title_2='', font_scale=1.2):
    con_mat_1 = confusion_matrix(labels=y_true_1, predictions=y_pred_1).numpy()
    con_mat_norm_1 = np.around(con_mat_1.astype('float') / con_mat_1.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df_1 = pd.DataFrame(con_mat_norm_1, index=class_names, columns=class_names)
    
    con_mat_2 = confusion_matrix(labels=y_true_2, predictions=y_pred_2).numpy()
    con_mat_norm_2 = np.around(con_mat_2.astype('float') / con_mat_2.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df_2 = pd.DataFrame(con_mat_norm_2, index=class_names, columns=class_names)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title, fontsize=16)
    
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mat_df_1, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax1)
    sns.heatmap(con_mat_df_2, annot=True, cmap=plt.cm.Blues, cbar=False, square=True, ax=ax2)
    
    ax1.set_title(sub_title_1)
    ax1.set_ylabel('True phase')
    ax1.set_xlabel('Predicted phase')
    ax2.set_title(sub_title_2)
    ax2.set_ylabel('True phase')
    ax2.set_xlabel('Predicted phase')
    
    plt.tight_layout(w_pad=4.0, h_pad=1.5)
    plt.show()