# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K

def sum_squared_error(y_true, y_pred):
    return (K.sum(K.square(y_pred - y_true))/2)

def ssim(y_true,y_pred):
    return tf.image.ssim(y_true,y_pred,1.0)