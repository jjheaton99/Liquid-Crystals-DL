#Wasserstein GAN with gradient penalty for generation random liquid crystal textures

import os
from os.path import exists, join

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, save_img
from keras.layers import Input, Reshape, Conv2D, Dense, LeakyReLU, UpSampling2D
from keras.layers import BatchNormalization, LayerNormalization, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
import keras.backend as K

from functools import partial

class WGAN_GP():
#credit to https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py 
#for the base class from which this is adapted
    def __init__(self, load_dir='', load_saved_models=True, batch_size=32):
        self.batch_size = batch_size
        self.img_width = 256
        self.img_height = 256
        self.channels = 1
        self.img_shape = (self.img_width, self.img_height, self.channels)
        self.latent_dim = 128
        self.history=[[], [], []]
        
        self.n_train_critic = 5
        optimiser = RMSprop(0.00005)
        
        models_loaded = False
        if load_saved_models:
            models_loaded = self.load_models(load_dir)           
            if models_loaded:
                print("models loaded successfully")

        if not models_loaded:
            if load_saved_models:
                print("failed to load models, building new ones")
            self.generator = self.build_generator()        
            self.critic = self.build_critic()
        
        #---critic setup---
        
        self.generator.trainable = False
        
        real_imgs = Input(shape=self.img_shape)
    
        z_discriminator = Input(shape=(self.latent_dim,))
        fake_imgs = self.generator(z_discriminator)
        
        fake = self.critic(fake_imgs)
        valid = self.critic(real_imgs)
        
        interpolated_imgs = self.get_interpolated_imgs(real_imgs, fake_imgs)
        valid_interpolated = self.critic(interpolated_imgs)
        
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_imgs)
        partial_gp_loss.__name__ = "gradient_penalty"
        
        self.critic_model = Model(inputs=[real_imgs, z_discriminator],
                                  outputs=[valid, fake, valid_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimiser,
                                  loss_weights=[1, 1, 10])
    
        #---generator setup---
        
        self.generator.trainable = True
        self.critic.trainable = False
        
        z_generator = Input(shape=(self.latent_dim,))
        imgs = self.generator(z_generator)
        valid = self.critic(imgs)
        
        self.generator_model = Model(z_generator, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimiser)
        
    def get_interpolated_imgs(self, real_imgs, fake_imgs):
        alpha = K.random_uniform(shape=(self.batch_size, 1, 1, 1))
        return alpha * real_imgs + (1 - alpha) * fake_imgs
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    #alternate gradient function, because sometimes keras.backend.gradients returns None
    def _compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]
    
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = self._compute_gradients(y_pred, [averaged_samples])[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_norm)
        return K.mean(gradient_penalty)
    
    def build_generator(self):
        model = Sequential([
            Dense(256 * 32 * 32, activation='relu', input_dim=self.latent_dim),
            Reshape((32, 32, 256)),
            UpSampling2D(),
            
            Conv2D(128, kernel_size=(4, 4), strides=1, activation='relu', padding='same'),
            BatchNormalization(momentum=0.8),
            UpSampling2D(),
            
            Conv2D(64, kernel_size=(4, 4), strides=1, activation='relu', padding='same'),
            BatchNormalization(momentum=0.8),
            UpSampling2D(),
            
            Conv2D(self.channels, kernel_size=(4, 4), activation='tanh', padding='same')
        ])
        
        model.summary()
        
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)
    
    def build_critic(self):
        model = Sequential([
            Conv2D(32, kernel_size=(4, 4), strides=2, input_shape=self.img_shape, padding='same'),
            LayerNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),
            
            Conv2D(64, kernel_size=(4, 4), strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),
            
            Conv2D(128, kernel_size=(4, 4), strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),
            
            Conv2D(256, kernel_size=(4, 4), strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),
            
            Conv2D(512, kernel_size=(4, 4), strides=2, padding='same'),
            LayerNormalization(),
            LeakyReLU(0.1),
            Dropout(0.2),
            
            Flatten(),
            Dense(1)
        ])
        
        model.summary()
        
        img = Input(shape=self.img_shape)
        validity = model(img)
        
        return Model(img, validity)
    
    def train(self, epochs, train_data, checkpoint_interval=100, sample_interval=100, 
              num_samples=10, history_plot_interval=0, save_dir='', verbose=True):
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))
        
        plot_history = False
        if history_plot_interval != 0:
            plot_history = True
        
        count = 0        
        for epoch in range(epochs):
            for _ in range(self.n_train_critic):
                count += 1
                #resets data generator if whole set has been iterated over
                if (train_data.n - (count * self.batch_size)) < self.batch_size:
                    train_data.reset()
                    count = 0
                    
                next_batch = train_data.next()
                #if else handles cases where data has labels or no labels
                if next_batch[0].shape == (self.batch_size,) + self.img_shape:
                    imgs = (next_batch[0] - 127.5) / 127.5
                else:
                    imgs = (next_batch - 127.5) / 127.5

                noise = np.random.normal(0.0, 1.0, (self.batch_size, self.latent_dim))
                discriminator_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                      [valid, fake, dummy])
                
            generator_loss = self.generator_model.train_on_batch(noise, valid)
            
            self.history[0].append(epoch)
            self.history[1].append(discriminator_loss[0])
            self.history[2].append(generator_loss)
            
            if epoch % sample_interval == 0:
                self.sample_images(epoch, num_samples, save_dir)
                
            if epoch % checkpoint_interval == 0:
                self.save_models(save_dir)
                
            if plot_history:
                if epoch % history_plot_interval == 0:
                    plt.close()
                    self.plot_history()
            
            if verbose:
                print("epoch %d/%d, D loss: %f, G loss: %f" % (epoch, epochs, discriminator_loss[0], generator_loss))
                
        return self.history
    
    def plot_history(self):
        fig, axis = plt.subplots(1)
        fig.suptitle('WGAN-GP training losses')
        
        axis.plot(self.history[1], label='critic_loss')
        axis.plot(self.history[2], label='gen_loss')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Loss')
        axis.legend(loc='lower left')
        
        plt.show()
            
    def sample_images(self, epoch=0, num_samples=10, save_dir=''):
        #check for/create the directory for saving samples
        sample_dir = ''
        if save_dir == '':
            sample_dir = 'samples'
            if not exists(sample_dir):
                os.makedirs(sample_dir)
        else:
            sample_dir = join(save_dir, 'samples')
            if not exists(sample_dir):
                os.makedirs(sample_dir)
        
        #generate and save samples
        noise = np.random.normal(0.0, 1.0, (num_samples, self.latent_dim))
        g_img_samples = (self.generator.predict(noise) * 127.5) + 127.5
        
        for sample in range(num_samples):
            save_img(join(sample_dir, 'epoch%d_sample%d.jpg' % (epoch, sample)), g_img_samples[sample])
        
    def save_models(self, save_dir=''):
        #check for/create directories for both models
        g_dir=''
        c_dir=''
        
        if save_dir == '':
            g_dir = 'generator'
            if not exists(g_dir):
                os.makedirs(g_dir)   
                
            c_dir = 'critic'
            if not exists(c_dir):
                os.makedirs(c_dir)
                
        else:
            g_dir = join(save_dir, 'generator')
            if not exists(g_dir):
                os.makedirs(g_dir)
                
            c_dir = join(save_dir, 'critic')
            if not exists(c_dir):
                os.makedirs(c_dir)
        
        #save models
        self.generator.save(g_dir, overwrite=True)
        self.critic.save(c_dir, overwrite=True)
    
    def load_models(self, load_dir=''):
        if load_dir == '':
            g_dir = 'generator'
            c_dir = 'critic'
            
        else:
            g_dir = join(load_dir, 'generator')
            c_dir = join(load_dir, 'critic')
            
        if exists(join(g_dir, 'saved_model.pb')) and exists(join(c_dir, 'saved_model.pb')):
            self.generator = load_model(g_dir, compile=False)
            self.critic = load_model(c_dir, compile=False)
            return True
        
        return False
        
if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True)
    
    train_data = train_datagen.flow_from_directory(
        directory='cholesteric/data',
        target_size=(256, 256),
        class_mode='categorical',
        color_mode='grayscale',
        batch_size=32,
        shuffle=True)
    
    wgan_gp = WGAN_GP(load_saved_models=False, batch_size=32)
    
    wgan_gp.train(epochs=99999999, train_data=train_data, checkpoint_interval=100, 
                  sample_interval=50, num_samples=10, history_plot_interval=10, save_dir='cholesteric')