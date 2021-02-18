import tensorflow as tf
from tensorflow.keras.layers import *

def simple_net(input_shape,hidden_shape,out_shape):
    """ Simple network """
    inp = Input(shape=input_shape)
    x = inp
    x = SimpleRNN(hidden_shape,activation='tanh')(x)
    out = Dense(out_shape,activation='softmax')(x)
    m = tf.keras.Model(inputs=inp,outputs=out)
    return m 
