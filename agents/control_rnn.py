import tensorflow as tf
from tensorflow.keras.layers import *
from agents.rnn_agent import RNNAgent
from nn.noisy_rnn import *
from nn.timestep_layers import *
from nn.separated_simple_rnn import *
from nn.custom_objects import *
import numpy as np
import os

class ControlRNN(RNNAgent):
    """ Agent using a RNN to solve the CONTROL task """
    def __init__(self,hidden_shape,input_shape=(None,4),out_shape=2,lr=10**-3,noise=None,random_training=False,activation='tanh'):
        self.activation = activation
        self.noise = noise
        self.random_training = random_training
        super().__init__(input_shape,hidden_shape,out_shape,lr=lr)
        self.postname = 'control'
        self.layer_list = ['rnn','dense']

    def parameters(self):
        """ Returns parameters of the networks """
        return self.nn.weights
        
    def reset_rnn(self):
        """ Define/Reset the RNN """
        #Model
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
        #Out network
        self.dense = Dense(self.out_shape,activation='softmax')

        #Build the model
        x = inp
        x = self.rnn(x)
        x = self.dense(x)
        self.nn = tf.keras.Model(inputs=inp,outputs=x)
    
    def forward(self,batch):
        """ Computes the output of the model from a Batch """
        x = batch.get('obs')
        x = self.rnn(x)
        x = self.timestep_chooser(x)
        x = self.dense(x)
        return x

class SeparatedControlRNN(ControlRNN):
    """ Agent using a separated RNN to solve the CONTROL task """
    def __init__(self,*args,separated_params=None,free_feedback=True,**kwargs):
        self.separated_params = separated_params
        self.free_feedback = free_feedback
        super().__init__(*args,**kwargs)
    def reset_rnn(self):
        #Model
        
        #Cell definition
        if self.noise:
            self.rnn_cell = NoisySeparatedSimpleRNNCell(self.hidden_shape,self.separated_params,free_feedback=self.free_feedback,activation=self.activation,noise=self.noise)
        else:
            self.rnn_cell = SeparatedSimpleRNNCell(self.hidden_shape,self.separated_params,free_feedback=free_feedback,activation=self.activation)
        self.rnn = RNN(self.rnn_cell,return_sequences=True)
        #Random Training
        if self.random_training:
            self.timestep_chooser = RandomTimestep()
        else:
            self.timestep_chooser = LastTimestep()
        #Out network
        self.dense = Dense(self.out_shape,activation='softmax')

        #Build the model
        inp = Input(shape=self.input_shape)
        x = inp
        x = self.rnn(x)
        x = self.dense(x)
        self.nn = tf.keras.Model(inputs=inp,outputs=x)
    
    def forward(self,batch):
        """ Compute the output of the model from a Batch """
        x = batch.get('obs')
        x = self.rnn(x)
        x = self.timestep_chooser(x)
        x = self.dense(x)
        return x
