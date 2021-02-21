#for splitting image data into training, validation and test sets

import os
import numpy as np
from image_data_transformer import load_images

def make_class_dirs(class_folders, output_path):
    for class_name in class_folders:
        folder = output_path + '/' + class_name
        if not os.path.isdir(folder):
            os.mkdir(output_path + '/' + class_name)

def create_and_save_sets(input_path, output_path, val_fraction=0.2, test_fraction=0.1):
    class_folders = os.listdir(input_path)
    train_folder = output_path + '/train'
    val_folder = output_path + '/valid'
    test_folder = output_path + '/test'
    
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    make_class_dirs(class_folders, train_folder)
    
    if not os.path.isdir(val_folder):
        os.mkdir(val_folder)
    make_class_dirs(class_folders, val_folder)
        
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
    make_class_dirs(class_folders, test_folder)
    
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
    create_and_save_sets('C:/MPhys project/Liquid-Crystals-DL/data/Images/Black and white', 
                         'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data')