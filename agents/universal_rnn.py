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
        self.random_timestep_chooser = RandomTimestep()
        self.last_timestep_chooser = LastTimestep()
            
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
    
    def forward(self,batch,training=False,return_states=False):
        """ Computes the output of the model from a Batch """
        #get ref
        batch_obs = batch.get('obs')[:,:,:]
        # -- context encoding
        state = tf.zeros((tf.shape(batch_obs)[0],self.hidden_shape),dtype=tf.float64)
        # -- sequence integration
        x = batch_obs
        x0 = x[:,:1,:]
        out0 = self.rnn(x0,initial_state=state)

        xseq = x[:,1:-1,:]
        outseq = self.rnn(xseq,initial_state=out0[:,-1,:])

        if training and self.random_training:
            outseq_timed = self.random_timestep_chooser(outseq)
        else:
            outseq_timed = self.last_timestep_chooser(outseq)

        xl = x[:,-1:,:]
        outl = self.rnn(xl,initial_state=outseq_timed)

        #x = self.rnn(x0,initial_state=state)
        #x = self.timestep_chooser(x)
        # -- probabilities computation
        out = self.out_dense(outl[:,0,:])
        if return_states:
            return out,tf.concat([out0,outseq,outl],axis=1)
        return out
    
#Universal RRN that take their decision at the last layer only
class LazyUniversalRNN(UniversalRNN):
    def lazy_reinforce_loss(self,batch):
        """ Computes the loss from a batch and labels """
        #Get probas for colors
        out_actions,out_probas = self.predict(batch,return_probas=True)
        #Tensorflow way to get proba associated with chosen colors
        indices = [[i,out_actions[i]] for i in range(out_actions.shape[0])]
        pact = tf.gather_nd(out_probas,indices)
        #Rewards computation
        rews = tf.cast(tf.equal(out_actions,batch.get('color')),dtype=tf.float64)*2-1
        #normalize rewards
        rews = rews - tf.reduce_mean(rews)
        #Loss computation
        l = rews*tf.math.log(pact)
        #print('--',l[:5])
        lmean = tf.reduce_mean(l)
        return -lmean