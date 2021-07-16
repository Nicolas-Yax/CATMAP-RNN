import tensorflow as tf
from tensorflow.keras.layers import *
from agents.rnn_agent import RNNAgent
from nn.noisy_rnn import *
from nn.timestep_layers import *
from nn.separated_simple_rnn import *
import numpy as np
import os

class PostRNN(RNNAgent):
    """ Agent using a RNN to solve the POST task """
    def __init__(self,hidden_shape,input_shape=(None,2),out_shape=2,context_input_shape=(52,),lr=10**-3,noise=None,random_training=False,activation='tanh'):
        self.activation = activation
        self.context_input_shape = context_input_shape
        self.noise = noise
        self.random_training = random_training
        super().__init__(input_shape,hidden_shape,out_shape,lr=lr)

        self.postname = 'post'
        self.layer_list = ['rnn','context_dense','out_dense']

    def parameters(self):
        """ Returns parameters of the networks """
        return [self.nn.weights,self.context_nn.weights,self.out_nn.weights]

    def reset_rnn(self):
        """ Define/Reset the RNN """
        self.reset_nn()
        self.reset_context()
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

    def reset_context(self):
        """ Define/Reset the context nn """
        #Context network
        inp = Input(shape=self.context_input_shape)
        self.context_dense = Dense(self.hidden_shape,activation=self.activation)
        
        #Build the model
        x = inp
        out = self.context_dense(x)
        self.context_nn = tf.keras.Model(inputs=inp,outputs=out)

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
        batch_obs = batch.get('obs')[:,:,:2]
        ref = batch.get('obs')[:,0,2:]
        # -- context encoding
        state = tf.zeros((tf.shape(batch_obs)[0],self.hidden_shape),dtype=tf.float64)
        # -- sequence integration
        x = batch_obs
        x = self.rnn(x,initial_state=state)
        x = self.timestep_chooser(x)
        # -- context encoding
        context_inp = tf.concat([x,ref],axis=1)
        x = self.context_nn(context_inp)
        # -- probabilities computation
        out = self.out_dense(x)
        
        return out

class SeparatedPostRNN(PostRNN):
    """ Agent using a separated RNN to solve the POST task """
    def __init__(self,*args,separated_params=None,free_feedback=True,input_shape=(None,4),**kwargs):
        self.separated_params = separated_params
        self.free_feedback = free_feedback
        super().__init__(*args,input_shape=input_shape,**kwargs)

        self.layer_list = ['rnn','dense']

    def reset_rnn(self):
        """ Define/Reset the rnn """
        self.reset_nn()

    def parameters(self):
        """ Return parameters of the network """
        return [self.nn.weights]

    def reset_nn(self):
        """ Define/Reset the nn """
        #Model
        self.inp = Input(shape=self.input_shape)
        #Cell definition
        if self.noise:
            self.rnn_cell = NoisySeparatedSimpleRNNCell(self.hidden_shape,self.separated_params,free_feedback=self.free_feedback,activation=self.activation,noise=self.noise)
        else:
            self.rnn_cell = SeparatedSimpleRNNCell(self.hidden_shape,self.separated_params,free_feedback=self.free_feedback,activation=self.activation)
        #RNN
        self.rnn = RNN(self.rnn_cell,return_sequences=True)
        #Random Training
        if self.random_training:
            self.timestep_chooser = RandomTimestep()
        else:
            self.timestep_chooser = LastTimestep()
        #Out network
        self.inter = Dense(15,activation=self.activation)
        self.dense = Dense(self.out_shape,activation='softmax')
        #Network build
        x = self.inp
        x = self.rnn(x)
        out = self.timestep_chooser(x)
        out = self.dense(out)
        out = out
        self.nn = tf.keras.Model(inputs=self.inp,outputs=[x,out])

    def forward(self,batch):
        """ Compute the output of the model from a Batch """
        x = batch.get('obs').copy()
        
        refs = x[:,[0],2:]
        zeros = tf.zeros(tf.shape(refs))
        refs = np.concatenate([zeros,refs],axis=2)
        x[:,:,2:] = 0
        
        x = self.rnn(x)
        x = self.timestep_chooser(x)
        x = self.rnn(refs,initial_state=[x])
        x = x[:,-1]
        x = self.dense(x)
        
        return x
