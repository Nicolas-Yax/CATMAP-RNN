import tensorflow as tf
from tensorflow.keras.layers import *
from agents.rnn_agent import RNNAgent
import numpy as np

class ControlRNN(RNNAgent):
    def __init__(self,hidden_shape,input_shape=(None,4),out_shape=2,lr=10**-3):
        super().__init__(input_shape,hidden_shape,out_shape,lr=lr)

    def reset_rnn(self):
        #Model
        inp = Input(shape=self.input_shape)
        x = inp
        x = SimpleRNN(self.hidden_shape,activation='tanh')(x)
        out = Dense(self.out_shape,activation='softmax')(x)
        m = tf.keras.Model(inputs=inp,outputs=out)
        self.nn = m
    
    def forward(self,batch):
        return self.nn(batch.get('obs'))
