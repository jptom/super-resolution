# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import numpy as np
import cv2 as cv
    
def get_srcnn():
    model = keras.Sequential()
    model.add(kl.Conv2D(filters=64, kernel_size=9, activation='relu', padding='same', input_shape=(None, None, 3)))
    model.add(kl.Conv2D(filters=32, kernel_size=1, activation='relu', padding='same'))
    model.add(kl.Conv2D(filters=3, kernel_size=5, activation='relu', padding='same'))
    model.summary()
    return model
    
def get_espcn(factor):
    model = keras.Sequential()
    model.add(kl.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=(None, None, 3)))
    model.add(kl.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(kl.Conv2D(filters=factor**2*3, kernel_size=3, padding='same'))
    model.add(kl.Lambda(lambda z: tf.nn.depth_to_space(z, factor)))
    model.summary()
    return model

def get_fsrcnn(factor):
    model = keras.Sequential()
    model.add(kl.Conv2D(filters=56, kernel_size=5, padding='same', input_shape=(None, None, 3)))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2D(filters=12, kernel_size=1, padding='same'))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2D(filters=12, kernel_size=3, padding='same'))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2D(filters=12, kernel_size=3, padding='same'))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2D(filters=12, kernel_size=3, padding='same'))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2D(filters=56, kernel_size=1, padding='same'))
    model.add(kl.PReLU(shared_axes=[1, 2]))
    model.add(kl.Conv2DTranspose(filters=3, kernel_size=9, strides=(factor, factor), padding='same'))
    model.summary()
    return model

class My_Sequential(keras.Sequential):
    def add_layers(self, layers):
        for layer in layers:
            self.add(layer)

def residual_block_layers(channel, v=0, batch_norm=True):
    if v == 0:
        if batch_norm:
            residual_block = [kl.Conv2D(channel, kernel_size=3, padding='same'),
                              kl.Conv2D()]

    residual_block = 1

def get_srResNet(factor):
    pass

def get_srgan(factor):
    pass

def get_vdsr(num_layers=20):
    '''
    class VDSR(keras.Model):
        def __init__(self, num_layers=20):
            super().__init__()
            self.num_layers=num_layers
            
            self.net = keras.Sequential()
            for i in range(num_layers-1):
                self.net.add(kl.Conv2D(filters=64, kernel_size=3, padding='same',activation='relu'))
            self.net.add(kl.Conv2D(filters=3, kernel_size=3, padding='same'))
        def call(self, lr):
            x = self.net(lr)
            return kl.add([x, lr])
        
        #def get_config(self):
        #    return {"num_layers": self.num_layers}

        #@classmethod
        #def from_config(cls, config):
        #    return cls(**config)
    
    model = VDSR(num_layers=num_layers)
    model.build(input_shape=(None, None, None, 3))
    '''
    
    lr = keras.layers.Input(shape=(None, None, 3))
    x = kl.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')(lr)
    for i in range(num_layers):
        x = kl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    residual = kl.Conv2D(filters=3, kernel_size=3, padding='same')(x)    
    hr = kl.add([residual, lr])
    
    model = keras.Model(inputs=lr, outputs=hr)
    model.summary()
    return model

class ScaleLayer(kl.Layer):
    def __init__(self, initializer=None, regularizer=None):
        super(ScaleLayer, self).__init__()
        self.initializer = initializer
        self.regularizer = regularizer
        
    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     initializer=self.initializer,
                                     regularizer=self.regularizer,
                                     trainable=True)
        super(ScaleLayer, self).build(input_shape)  
    
    def call(self, inputs):
        return inputs * self.scale
    
    def get_config(self):
        return {"initializer": self.initializer,
                "regularizer": self.regularizer}
    

   

    
def get_drcn(num_recurrence=16):
    
    regularizer = tf.keras.regularizers.L2(0.01)
    
    lr = keras.layers.Input(shape=(None, None, 3))
    
    # embedding network
    x = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(lr)
    x = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        
        
    # inferece network
    outputs = []
    W = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
    for i in range(num_recurrence):
        x = W(x)
        outputs.append(x)
    
    # reconstruction newtwork
    W1 = kl.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)
    
    initializer = keras.initializers.Constant(1/num_recurrence)
    W2 = [ScaleLayer(initializer=initializer, regularizer=regularizer) for i in range(num_recurrence)]

    for i in range(num_recurrence):
        outputs[i] = W2[i](W1(outputs[i]) + lr)
    hr = kl.add(outputs)
    
    model = keras.Model(inputs=lr, outputs=hr)
    model.summary()
    return model

if __name__ == "__main__":
    import os
    
    '''
    # espcn test
    model = get_espcn(3)
    for i in range(17, 26):
        input_img = kl.Input(shape=(i, i, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
        
    # fsrcnn test
    model = get_fsrcnn(3)
    for i in range(17, 26):
        input_img = kl.Input(shape=(i, i, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
    
    model = get_vdsr(20)
    for i in range(17, 26):
        input_img = kl.Input(shape=(i*3, i*3, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
    ''' 
    model = get_drcn(16)
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    for i in range(17, 26):
        input_img = kl.Input(shape=(i*3, i*3, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
`
    
    if os.path.exists("weight.h5"):
        os.remove("weights.h5")
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    
    
    