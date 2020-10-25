import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import confusion_matrix

def display_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    con_mat = confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=class_names, columns=class_names)
    
    figsize=(len(class_names), len(class_names))
    figure = plt.figure(figsize=figsize)
    plt.suptitle(title)
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()