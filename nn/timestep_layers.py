import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

""" Layers used to handle random training """

class RandomTimestep(Layer):
    """ Takes an input of shape (X,Y,Z) and returns a (X,Z) cut of the input volume selecting a random y for Y axis """
    def __init__(self):
        super().__init__()
    def get_indexs(self,nb,minv=0,maxv=12):
        rangeind = tf.range(nb)
        randomind = tf.random.uniform((nb,),minval=minv,maxval=maxv,dtype=tf.int32)
        ind = tf.concat([rangeind[:,None],randomind[:,None]],axis=1)
        return ind
    def call(self,inputs):
        x = inputs
        ind = self.get_indexs(tf.shape(x)[0],maxv=tf.shape(x)[1])
        out = tf.gather_nd(x,ind)
        return out

class LastTimestep(Layer):
    """ Takes an input of shape (X,Y,Z) and returns the last (X,Z) cut of the sequence on Y """
    def call(self,inputs):
        out = inputs[:,-1,:]
        return out
