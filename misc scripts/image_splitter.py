#For dividing up large texture images into smaller squares

from os.path import join

import numpy as np
from PIL import Image

from image_data_transformer import load_images

#splits given image into specified number of rows and columns and returns list of new images
def split_image(image, rows, columns):
    img = np.array(image)
    
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    channels = np.shape(img)[2]
    
    new_width = width//columns
    new_height = height//rows
        
    new_imgs = np.empty((rows * columns, new_height, new_width, channels))
    
    index = 0
    for row in range(rows):
        for column in range(columns):
            slice_row = row * new_height
            slice_col = column * new_width
            new_imgs[index] = img[slice_row:slice_row+new_height, slice_col:slice_col+new_width]
            index += 1
            
    return [Image.fromarray(new_img.astype(np.uint8)) for new_img in new_imgs]
    
def split_and_save_dir(input_path, output_path, file_name, rows, columns):
    imgs = load_images(input_path)
    
    count = 0
    for img in imgs:
        for new_img in split_image(img, rows, columns):
            count += 1
            new_img.save(join(output_path, file_name + '%d.png' % count))
            
if __name__ == '__main__':
    split_and_save_dir(
        'D:/MPhys project/Liquid-Crystals-DL/data/Images/Colour unedited/temp for split', 
        'D:/MPhys project/Liquid-Crystals-DL/data/Images/Colour unedited/temp for sort',
        file_name = 't',
        rows=2, 
        columns=3)