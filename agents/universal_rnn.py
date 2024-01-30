import tensorflow as tf
from tensorflow.keras.layers import *
from agents.rnn_agent import RNNAgent
from nn.noisy_rnn import *
from nn.timestep_layers import *
from nn.separated_simple_rnn import *
import numpy as np
import os

class UniversalRNN(RNNAgent):
    """ Agent using a RNN to solve the POST task """
    def __init__(self,hidden_shape,input_shape=(None,4),out_shape=2,lr=10**-3,noise=None,random_training=False,activation='tanh'):
        self.activation = activation
        self.noise = noise
        self.random_training = random_training
        super().__init__(input_shape,hidden_shape,out_shape,lr=lr)

        self.postname = 'post'
        self.layer_list = ['nn','out_nn']

    def parameters(self):
        """ Returns parameters of the networks """
        return self.nn.weights + self.out_nn.weights #Concat lists of params
    
    def reset_rnn(self):
        """ Define/Reset the RNN """
        self.reset_nn()
        self.reset_out()

    def reset_nn(self):
        """ Define/Reset the nn """
        #Classic network
        inp = Input(shape=self.input_shape)
        #Cell definition
        self.rnn_cell = SimpleRNNCell(self.hidden_shape,activation=self.activation)
        #Wrap cell
        if self.noise:
            self.rnn_cell = noisy_cell_wrap(self.rnn_cell,self.noise)
        #RNN
        self.rnn = RNN(self.rnn_cell,return_sequences=True)
        #Random Training
        if self.random_training:
            self.timestep_chooser = RandomTimestep()
        else:
            self.timestep_chooser = LastTimestep()
            
        #Build the model
        x = inp
        out = self.rnn(x)
        m = tf.keras.Model(inputs=inp,outputs=out)
        self.nn = m
        
    def reset_out(self):
        """ Define/Reset the out nn """
        #Out network
        inp = Input(shape=self.hidden_shape)
        self.out_dense = Dense(self.out_shape,activation='softmax')
        #Build the model
        x = inp
        out = self.out_dense(x)
        self.out_nn = tf.keras.Model(inputs=inp,outputs=out)
    
    def forward(self,batch):
        """ Computes the output of the model from a Batch """
        #get ref
        batch_obs = batch.get('obs')[:,:,:]
        # -- context encoding
        state = tf.zeros((tf.shape(batch_obs)[0],self.hidden_shape),dtype=tf.float64)
        # -- sequence integration
        x = batch_obs
        x = self.rnn(x,initial_state=state)
        x = self.timestep_chooser(x)
        # -- probabilities computation
        out = self.out_dense(x)
        
        return out