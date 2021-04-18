from os.path import join, exists, isdir
from os import listdir
from os import mkdir

import numpy as np

from PIL import Image

from skimage.color import rgb2gray
from skimage.transform import resize

def load_images(path):
    #lists all files not including directories
    img_list = [file for file in listdir(path) if isdir(join(path, file)) == False]
    loaded_imgs = []
    
    for image in img_list:
        img = Image.open(join(path, image))
        loaded_imgs.append(img)
        
    return loaded_imgs

#crops images to square and resizes to given size
def transform_image(image, as_array=False, size=256, black_and_white=False):
    img = np.array(image)

    height = np.shape(img)[0]
    width = np.shape(img)[1]
    square = (height == width)
    
    #if image is not square, crop excess height/width
    if not square:
        diff = height - width
        crop = int(np.abs(diff) / 2)
        
        if diff > 0:
            img = img[crop:width+crop, :]
        else:
            img = img[:, crop:height+crop]
    
    if black_and_white:
        img = rgb2gray(img)
        
    img = resize(img, (size, size), anti_aliasing=False)
    img *= 255
    img = img.astype(np.uint8)
    
    if as_array:
        return img
    
    return Image.fromarray(img)

def transform_path_images(input_path, output_path, size, black_and_white):
    #transforms images not in sub folders
    imgs = load_images(input_path)
    count = 0
    for image in imgs:
        count += 1
        t_img = transform_image(image, False, size, black_and_white)
        t_img.save(join(output_path, '%d.png' % count))

#applies transformation to entire directory of images
#if nested_dirs is true, checks for sub folders of images
def transform_directory(input_path, output_path, nested_dirs=False, size=256, black_and_white=False):
    #transforms any images not in folders
    transform_path_images(input_path, output_path, size, black_and_white)
    
    #create list of all sub directories in input path    
    if nested_dirs:
        sub_dirs = [sub_dir for sub_dir in listdir(input_path) if isdir(join(input_path, sub_dir)) == True]
        for sub_dir in sub_dirs:
            #ensures dirs in the input path are also in the output path
            if not exists(join(output_path, sub_dir)):
                mkdir(join(output_path, sub_dir))
                
            #transforms and copies images between paths
            transform_path_images(join(input_path, sub_dir),
                                  join(output_path, sub_dir),
                                  size,
                                  black_and_white)

if __name__ == '__main__':

    transform_directory('C:/MPhys project/Liquid-Crystals-DL/data/Images/Colour unedited/IF2/train', 
                        'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/train',
                        nested_dirs=True)
    
    transform_directory('C:/MPhys project/Liquid-Crystals-DL/data/Images/Colour unedited/IF2/valid', 
                        'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/valid',
                        nested_dirs=True)
    
    transform_directory('C:/MPhys project/Liquid-Crystals-DL/data/Images/Colour unedited/IF2/test', 
                        'C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/IF2/test',
                        nested_dirs=True)