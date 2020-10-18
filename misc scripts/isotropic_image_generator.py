#random dark noise grayscale image generator for isotropic class

import numpy as np

from PIL import Image

def generate_image(size, max_brightness=30):
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            img[i][j] = np.random.randint(0, max_brightness, dtype=np.uint8)
        
    return img

def create_and_save_images(path, size=200, max_brightness=30, quantity=500):
    for count in range(quantity):
        img = Image.fromarray(generate_image(size, max_brightness))
        img = img.convert('L')
        img.save(path + '/' + '%d.png' % count)

if __name__ == '__main__':
    create_and_save_images('D:/MPhys project/Liquid-Crystals-DL/Data/Images/Black and white/isotropic')