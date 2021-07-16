import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

def noisy_cell_wrap(cell,sigma):
    """ Noisy wrap of SimpleRNNCell (doesn't implement the noise for separated net paradigm) """
    sigma = tf.convert_to_tensor(sigma,dtype=tf.float64)
    #Save original call method
    prec_call = cell.call
    #Wrap of call function which will overload the SimpleRNNCell call method
    def call(*args,**kwargs):
        out,state = prec_call(*args,**kwargs)
        state = state[0]
        noisy_out = tf.random.normal(shape=tf.shape(out),stddev=sigma,dtype=tf.float64)+out
        noisy_state = tf.random.normal(shape=tf.shape(state),stddev=sigma,dtype=tf.float64)+state
        noisy_state = [noisy_state]
        return (noisy_out,noisy_state)
    #Overide
    cell.call = call
    return cell
