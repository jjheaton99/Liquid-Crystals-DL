#for splitting image data into training, validation and test sets

import os
import numpy as np
from image_data_transformer import load_images

def make_class_dirs(class_folders, output_path):
    for class_name in class_folders:
        os.mkdir(output_path + '/' + class_name)

def create_and_save_sets(input_path, output_path, val_fraction=0.2, test_fraction=0.1):
    class_folders = os.listdir(input_path)
    
    os.mkdir(output_path + '/train')
    make_class_dirs(class_folders, output_path + '/train')
    
    os.mkdir(output_path + '/valid')
    make_class_dirs(class_folders, output_path + '/valid')
        
    os.mkdir(output_path + '/test')
    make_class_dirs(class_folders, output_path + '/test')
    
    for class_name in class_folders:
        train_imgs = load_images(input_path + '/' + class_name)
        val_imgs = []
        test_imgs = []
        
        img_count = len(train_imgs)
        val_count = int(val_fraction * img_count)
        test_count = int(test_fraction * img_count)
        
        for vals in range(val_count):
            rand_index = np.random.randint(len(train_imgs))
            val_imgs.append(train_imgs[rand_index])
            del train_imgs[rand_index]
            
        for tests in range(test_count):
            rand_index = np.random.randint(len(train_imgs))
            test_imgs.append(train_imgs[rand_index])
            del train_imgs[rand_index]
        
        count = 0
        for img in train_imgs:
            img.save(output_path + '/train/' + class_name + '/%d.png' % count)
            count += 1
            
        for img in val_imgs:
            img.save(output_path + '/valid/' + class_name + '/%d.png' % count)
            count += 1
            
        for img in test_imgs:
            img.save(output_path + '/test/' + class_name + '/%d.png' % count)
            count += 1

if __name__ == '__main__':
    create_and_save_sets('D:/MPhys project/Liquid-Crystals-DL/data/Images/Black and white', 
                         'D:/MPhys project/Liquid-Crystals-DL/data/Prepared data')