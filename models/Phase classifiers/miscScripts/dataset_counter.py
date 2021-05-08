import numpy as np
import pandas as pd
from os import listdir
from os.path import join

def count_images(dataset_path):
    classes = listdir(join(dataset_path, 'train'))

    image_counts = np.empty((3, len(classes)))
    
    for idx, name in enumerate(classes):
        image_counts[0][idx] = len(listdir(join(dataset_path, 'train', name)))
        image_counts[1][idx] = len(listdir(join(dataset_path, 'valid', name)))
        image_counts[2][idx] = len(listdir(join(dataset_path, 'test', name)))
        
    return image_counts, classes

def save_image_counts(dataset_path, save_path, save_name):
    image_counts, classes = count_images(dataset_path)
    image_counts = np.append(image_counts, np.expand_dims(np.sum(image_counts, axis=1), axis=1), axis=1)
    image_counts = np.append(image_counts, np.expand_dims(np.sum(image_counts, axis=0), axis=0), axis=0)
    
    rows = ['Training', 'Validation', 'Test', 'Totals']
    classes.append('Totals')
    
    pd.DataFrame(image_counts, columns=classes, index=rows).to_csv(join(save_path, save_name + '.csv'))
    
save_image_counts('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChSm', 
                  'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/dataset_counts', 
                  'ChSm')

save_image_counts('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/AC', 
                  'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/dataset_counts', 
                  'AC')

save_image_counts('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF', 
                  'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/dataset_counts', 
                  'IF')

save_image_counts('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChFluHex', 
                  'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/dataset_counts', 
                  'ChFluHex')

save_image_counts('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChACIF', 
                  'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/dataset_counts', 
                  'ChACIF')    

    
    
