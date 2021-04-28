from os.path import join, exists
from os import mkdir

import numpy as np
from PIL import Image

from con_mat_plotters import create_test_gen, get_labels_and_preds

def save_img_preds_ChACIF(model_dir, model_name, gen, valid=False):
    if valid:
        path = join('saved preds/ChACIF', model_name + 'VALID')
    else:
        path = join('saved preds/ChACIF', model_name + 'TEST')
    if not exists(path):
        mkdir(path)
    
    labels, preds = get_labels_and_preds(model_dir, gen)
    
    imgs = gen.next()[0]

    for idx, img in enumerate(imgs):
        if labels[idx] == 0:
            label = 'Ch'
        elif labels[idx] == 1:
            label = 'A'
        elif labels[idx] == 2:
            label = 'C'
        elif labels[idx] == 3:
            label = 'I'
        elif labels[idx] == 4:
            label = 'F'
        
        if preds[idx] == 0:
            pred = 'Ch'
        elif preds[idx] == 1:
            pred = 'A'
        elif preds[idx] == 2:
            pred = 'C'
        elif preds[idx] == 3:
            pred = 'I'
        elif preds[idx] == 4:
            pred = 'F'
    
        img = (img * 255).astype(np.uint8).reshape((256, 256))
        img = Image.fromarray(img)
        
        img.save(join(path, str(idx) + '_' + label + '_' + pred + '.png'))    

valid_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChACIF/valid')
test_gen = create_test_gen('C:/MPhys project/Liquid-Crystals-DL/data/Prepared data/ChACIF/test')

save_img_preds_ChACIF('checkpoints/ChACIF/sequential/seq_4_32_batch16_lr1e-4_a', 
                      'seq_4_32_batch16_lr1e-4_a', 
                      valid_gen,
                      True)

save_img_preds_ChACIF('checkpoints/ChACIF/sequential/seq_4_32_batch16_lr1e-4_a', 
                      'seq_4_32_batch16_lr1e-4_a', 
                      test_gen,
                      False)

save_img_preds_ChACIF('checkpoints/ChACIF/inception/inc_2_4_batch16_lr1e-4_a', 
                      'inc_2_4_batch16_lr1e-4_a', 
                      valid_gen,
                      True)

save_img_preds_ChACIF('checkpoints/ChACIF/inception/inc_2_4_batch16_lr1e-4_a', 
                      'inc_2_4_batch16_lr1e-4_a', 
                      test_gen,
                      False)


