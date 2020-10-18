from os import listdir

import numpy as np

from PIL import Image

from skimage.color import rgb2gray
from skimage.transform import resize

def load_images(path):
    
    img_list = listdir(path)
    loaded_imgs = []
    
    for image in img_list:
        img = Image.open(path + '/' + image)
        loaded_imgs.append(img)
        
    return loaded_imgs

def transform_image(image, size=200, black_and_white=True):
    
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
    
    return Image.fromarray(img)

def transform_directory(input_path, output_path, size=200, black_and_white=True):
    
    imgs = load_images(input_path)
    count = 0
    
    for image in imgs:
        count += 1
        t_img = transform_image(image, size, black_and_white)
        t_img.save(output_path + '/' + '%d.png' % count)

transform_directory('Data/Images/Colour unedited/cholesteric', 
                    'Data/Images/Black and white/cholesteric')

transform_directory('Data/Images/Colour unedited/columnar', 
                    'Data/Images/Black and white/columnar')

transform_directory('Data/Images/Colour unedited/nematic', 
                    'Data/Images/Black and white/nematic')

transform_directory('Data/Images/Colour unedited/smectic', 
                    'Data/Images/Black and white/smectic')

transform_directory('Data/Images/Colour unedited/twist_grain_boundary', 
                    'Data/Images/Black and white/twist_grain_boundary')