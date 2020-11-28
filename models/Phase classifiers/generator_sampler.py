import numpy as np
from PIL import Image

from sequential import train_gen

#for sampling augmented images from a data generator
def display_image_sample(num_images=1, generator=train_gen):
    for _ in range(num_images):
        batch = generator.next()[0]
        img = batch[np.random.randint(np.size(batch, axis=0))]
        img = np.add.reduce(img, axis=2)
        img *= 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.show()

display_image_sample(20)