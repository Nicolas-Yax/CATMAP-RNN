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
    def __init__(self,hidden_shape,input_shape=(None,4),out_shape=2,lr=10**-3,noise=None,activation='tanh'):
        self.activation = activation
        self.noise = noise
        self.random_training = True
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
        #concat out(x) and 1-out(x)
        out = self.out_dense(x)
        #out_ = tf.concat([out,1-out],axis=1)
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

        if True:#training and self.random_training:
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
            return out,tf.concat([out0[:,:],outseq[:,:],outl[:,:]],axis=1)
        return out
    
#Universal RRN that take their decision at the last layer only
class LazyUniversalRNN(UniversalRNN):

    def reset_opt(self):
        self.opt = tf.keras.optimizers.Adam(self.lr)
        self.adv_opt = tf.keras.optimizers.Adam(self.lr)

    def parameters(self):
        """ Returns parameters of the networks """
        return self.nn.weights + self.out_nn.weights #Concat lists of params
    
    def adv_parameters(self):
        """ Returns parameters of the networks """
        return self.adv_nn.weights

    def reset_rnn(self):
        """ Define/Reset the RNN """
        self.reset_nn()
        self.reset_out()
        self.reset_adv()

    def reset_adv(self):
        """ Define/Reset the adv nn """
        #Out network
        inp = Input(shape=self.hidden_shape)
        self.adv_dense = Dense(1,activation='sigmoid')
        #Build the model
        x = inp
        out = self.adv_dense(x)
        self.adv_nn = tf.keras.Model(inputs=inp,outputs=out)

    def lazy_reinforce_loss(self,batch):
        """ Computes the loss from a batch and labels """
        #Get probas for colors
        out_actions,out_probas,out_states = self.predict(batch,return_probas=True,return_states=True)
        #Try to predict decision from latent space
        out_states_flat = tf.reshape(out_states,(out_states.shape[0]*out_states.shape[1],)+(out_states.shape[2],))
        adv_choice_prediction_flat = self.adv_nn(out_states_flat)
        #Add the second prediction as 1-p
        adv_choice_prediction_flat_p = tf.concat([adv_choice_prediction_flat,1-adv_choice_prediction_flat],axis=1)
        #repeat out_actions over adv_choice_prediction steps
        out_actions = tf.repeat(out_actions,out_states.shape[1],axis=0)
        #Compute Cross-Entropy loss with actual out_actions
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(out_actions,adv_choice_prediction_flat_p)
        return tf.reduce_mean(ce_loss)
    
    def adv_loss(self,batch,i=None):
        return self.lazy_reinforce_loss(batch)
    
    def loss(self,batch,i=None):
        reinforce_loss = self.reinforce_loss(batch)
        adv_loss = self.adv_loss(batch)
        return reinforce_loss - adv_loss
    
    def fit(self,batch,nb_fit=5):
        """ Fit once the model and given batch and labels """
        for i in range(nb_fit):
            self.opt.minimize(lambda : self.loss(batch,i=i),var_list=self.parameters())
            self.adv_opt.minimize(lambda : self.adv_loss(batch,i=i),var_list=self.adv_parameters())