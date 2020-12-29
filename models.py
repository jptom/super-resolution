# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
import tensorflow.keras.layers as kl
    
def get_srcnn():
    model = keras.Sequential()
    model.add(kl.Conv2D(filters=64, kernel_size=9, activation='relu', padding='same', input_shape=(None, None, 3)))
    model.add(kl.Conv2D(filters=32, kernel_size=1, activation='relu', padding='same'))
    model.add(kl.Conv2D(filters=3, kernel_size=5,  activation='relu', padding='same'))
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

def residual_block(x, filters, batch_norm=True, scaling=False, activation=kl.ReLU()):
    residual = kl.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
    if batch_norm:
        residual = kl.BatchNormalization()(residual)
    residual = activation(residual)
    residual = kl.Conv2D(filters=filters, kernel_size=3, padding='same')(residual)
    if batch_norm:
        residual = kl.BatchNormalization()(residual)
    if type(scaling)==float:
        residual *= scaling
        
    return residual + x

def get_srResNet(factor, filters=64, num_resblocks=16):
    lr = kl.Input(shape=(None, None, 3))
    x = kl.Conv2D(filters=filters, kernel_size=9, padding='same')(lr)
    x = kl.PReLU(shared_axes=[1, 2])(x)
    skip = tf.identity(x)
    for i in range(num_resblocks):
        x = residual_block(x, filters, activation=kl.PReLU(shared_axes=[1, 2]))
    
    x += skip
    if factor == 3:
        x = kl.Conv2D(filters=factor**2*3, kernel_size=3, padding='same')(x)
        x = kl.Lambda(lambda z: tf.nn.depth_to_space(z, factor))(x)
    
    x = kl.Conv2D(filters=3, kernel_size=3, padding='same')(x)
    model = keras.Model(inputs=lr, outputs=x)
    model.summary()
    return model
    
def get_edsr(factor, filters=256, num_resblocks=32):
    lr = kl.Input(shape=(None, None, 3))
    x = kl.Conv2D(filters=filters, kernel_size=9, padding='same')(lr)
    skip = tf.identity(x)
    for i in range(num_resblocks):
        x = residual_block(x, filters, batch_norm=False, scaling=True)
        
    x += skip
    if factor == 3:
        x = kl.Conv2D(filters=factor**2*3, kernel_size=3, padding='same')(x)
        x = kl.Lambda(lambda z: tf.nn.depth_to_space(z, factor))(x)
    
    x = kl.Conv2D(filters=3, kernel_size=3, padding='same')(x)
    model = keras.Model(inputs=lr, outputs=x)
    model.summary()
    return model

def conv_block(x, filters, kernel_size, strides, activation=kl.ReLU()):
    x = kl.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = activation(x)
    
    return x

def get_discriminator():
    img = kl.Input(shape=(96, 96, 3))
    x = kl.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(img)        
    x = kl.LeakyReLU(alpha=0.2)(x)
    x = conv_block(x, 64, 3, 2, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 128, 3, 1, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 128, 3, 2, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 256, 3, 1, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 256, 3, 2, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 512, 3, 1, activation=kl.LeakyReLU(alpha=0.2))
    x = conv_block(x, 512, 3, 2, activation=kl.LeakyReLU(alpha=0.2))
    x = kl.Flatten()(x)
    x = kl.Dense(units=1024)(x)
    x = kl.LeakyReLU(alpha=0.2)(x)
    x = kl.Dense(units=1, activation="sigmoid")(x)
    
    model = keras.Model(inputs=img, outputs=x)
    model.summary()
    return model
    
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
    
    lr = kl.Input(shape=(None, None, 3))
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
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return {"initializer": self.initializer,
                "regularizer": self.regularizer}
    
def get_drcn(num_recurrence=16):
    
    regularizer = keras.regularizers.L2(0.0001)
    
    lr = keras.layers.Input(shape=(None, None, 3))
    
    # embedding network
    x = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer)(lr)
    x = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer)(x)
        
        
    # inferece network
    outputs = []
    W = kl.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=regularizer)
    for i in range(num_recurrence):
        x = W(x)
        outputs.append(x)
    
    # reconstruction newtwork
    W1 = kl.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                   kernel_regularizer=regularizer)
    '''
    initializer = keras.initializers.Constant(1/num_recurrence)
    W2 = [ScaleLayer(initializer=initializer, regularizer=None) for i in range(num_recurrence)]
    for i in range(num_recurrence):
        outputs[i] = W2[i](W1(outputs[i]) + lr)
    '''
    W2 = [kb.variable(1/num_recurrence) for i in range(num_recurrence)]
    for i in range(num_recurrence):
        outputs[i] = W2[i]*(W1(outputs[i]) + lr)
    hr = kl.add(outputs)
    
    model = keras.Model(inputs=lr, outputs=hr)
    model.summary()
    return model


def get_model(name, mag):
    if name == "srcnn":
        model = get_srcnn()
        
    elif name == "espcn":
        model = get_espcn(mag)
        
    elif name == "fsrcnn":
        model = get_fsrcnn(mag)
        
    elif name == "vdsr":
        model = get_vdsr()
            
    elif name == "drcn":
        model = get_drcn()
    
    elif name == "srresnet":
        model == get_srResNet(mag)
        
    elif name == "edsr":
        model = get_edsr(mag)
            
    else:
        raise ValueError("select correct model")
        
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
    
    model = get_drcn(16)
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    for i in range(17, 26):
        input_img = kl.Input(shape=(i*3, i*3, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
    model.save("model.h5")
    model = km.load_model("model.h5")
    
    if os.path.exists("weight.h5"):
        os.remove("weights.h5")
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    
    model = get_edsr(3)
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    for i in range(17, 26):
        input_img = kl.Input(shape=(i, i, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
    model.save("model.h5")
    model = km.load_model("model.h5")
    
    if os.path.exists("weight.h5"):
        os.remove("weights.h5")
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    
    model = get_srResNet(3)
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    for i in range(17, 26):
        input_img = kl.Input(shape=(i, i, 3))
        output = model(input_img)
        assert output.shape[1] == i*3
        assert output.shape[2] == i*3
    model.save("model.h5")
    model = km.load_model("model.h5")
    
    if os.path.exists("weight.h5"):
        os.remove("weights.h5")
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    
    model = get_discriminator()
    model.save_weights("weights.h5")
    model.load_weights("weights.h5")
    input_img = kl.Input(shape=(96, 96, 3))
    output = model(input_img)
    assert output.shape[1] == 1
    model.save("model.h5")
    model = km.load_model("model.h5")
    
    if os.path.exists("weight.h5"):
        os.remove("weights.h5")
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    '''
    