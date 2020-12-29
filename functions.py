# -*- coding: utf-8 -*
import tensorflow.keras as keras
import tensorflow.keras.backend as kb
keras.applications.EfficientNetB0()

def psnr(y_true, y_pred):
    return -10*kb.log(kb.mean(kb.flatten((y_true - y_pred))**2))/kb.log(10.0)