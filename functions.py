# -*- coding: utf-8 -*
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
keras.applications.EfficientNetB0()

class PSNR:
    def __init__(self, r):
        if r == None:
            self.r = 1
        else:
            self.r = (r[1]-r[0])**2
    
    def __call__(self, y_true, y_pred):
        return 10*kb.log(self.r/kb.mean(kb.flatten((y_true - y_pred))**2))/kb.log(10.0)