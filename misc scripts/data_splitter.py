#for splitting image data into training, validation and test sets

import os

from PIL import Image

def make_class_dirs(input_path, output_path):
    class_folders = os.listdir(input_path)
    for name in class_folders:
        os.mkdir(output_path + '/' + name)

def create_and_save_sets(input_path, output_path):
    os.mkdir(output_path + '/train')
    make_class_dirs(input_path)
    
    os.mkdir(output_path + '/valid')
    os.mkdir(output_path + '/test')
    
    