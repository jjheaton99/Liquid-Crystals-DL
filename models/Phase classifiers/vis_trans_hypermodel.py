import tensorflow as tf
import keras
from kerastuner import HyperModel

import vision_transformer

class VisTransHypermodel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build(self, hp):
        patch_dim_hp = hp.Int('patch_dim',
                              min_value=16,
                              max_value=32,
                              step=16,
                              default=16)
        model_dim_hp = hp.Int('model_dim',
                              min_value=32,
                              max_value=128,
                              step=32,
                              default=32)
        num_encoders_hp = hp.Int('num_encoders',
                                 min_value=4,
                                 max_value=8,
                                 step=4,
                                 default=4)    
        dropout_rate_hp = hp.Float('dropout_rate',
                                   min_value=0.0,
                                   max_value=0.1,
                                   step=0.1,
                                   default=0.0)
        lr_hp = hp.Float('learning_rate',
                         min_value=1e-4,
                         max_value=1e-2,
                         sampling='LOG',
                         default=1e-3)
        
        model = vision_transformer.VisionTransformer(input_shape=self.input_shape,
                                                     num_classes=self.num_classes,
                                                     patch_dim=patch_dim_hp,
                                                     model_dim=model_dim_hp,
                                                     num_encoders=num_encoders_hp,
                                                     dropout_rate=dropout_rate_hp)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_hp),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics='accuracy')
        
        return model