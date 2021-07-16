import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as backend
import numpy as np

class SeparatedSimpleRNNCell(SimpleRNNCell):
    """ Separated RNN Cell : implements a single timestep computation of a separated rnn paradygm """
    def __init__(self,units,params,free_feedback=True,*args,**kwargs):
        self.nb_separated = params[0] #Nb neurons which will get 1st input
        self.size_input = params[1] #Size of the 1st input
        self.free_feedback = free_feedback
        super().__init__(units+self.nb_separated,*args,**kwargs)
        
    def compute_kernel_mask(self):
        """ Input kernel mask """
        self.mask = np.zeros(self.kernel.shape)
        self.mask[:self.size_input,:self.nb_separated] = 1
        self.mask[self.size_input:,self.nb_separated:] = 1
        self.mask = 1-self.mask

    def compute_rec_kernel_mask(self):
        """ Recurrent kernel mask (for REF->SEQ only) """
        self.rec_mask = np.zeros(self.recurrent_kernel.shape)
        self.rec_mask[:self.nb_separated,:self.nb_separated] = 1
        self.rec_mask[:,self.nb_separated:] = 1
        #self.rec_mask = 1-self.rec_mask

    #----- Tensorflow spagetti code copied from tensorflow github SimpleRNNCell
    def call1(self,inputs,states,training=None):
        self.compute_kernel_mask()
        if not(self.free_feedback):
            self.compute_rec_kernel_mask()
        h = backend.dot(inputs, self.kernel*self.mask) #Apply the separated input mask
        h = backend.bias_add(h, self.bias)
        if not(self.free_feedback): #If limited feedbacks -> apply the recurrent mask
            output = h + backend.dot(states[0], self.recurrent_kernel*self.rec_mask)
        else:
            output = h + backend.dot(states[0], self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def call(self, inputs, states, training=None):
        output = self.call1(inputs,states,training=training)
        return output, [output]

    def get_config(self):
        c = super().get_config()
        c['params'] = (self.nb_separated,self.size_input)
        return c
    @classmethod
    def from_config(cls,config):
        cls(**config)
    #-----

class NoisySeparatedSimpleRNNCell(SeparatedSimpleRNNCell):
    """ Noisy cell for separated paradygm : applies noise in a part only if the input isn't a vector full of zeros """
    def __init__(self,*args,noise=None,**kwargs):
        assert noise != None
        self.noise = noise
        super().__init__(*args,**kwargs)

    #Overide of previous tensorflow spagetti code
    def call1(self,inputs,states,training=None):
        output = super().call1(inputs,states,training=training)
        
        #Noise vector (only few values will be used depending on zeros in the input
        noise_vector = tf.random.normal(tf.shape(output),stddev=self.noise,dtype=tf.float64)
        
        #Compute the mask for where to apply noise
        #-- Part 1 of the network
        index_vector1 = tf.reduce_any(inputs[:,:self.size_input]!=0,axis=1)
        index_vector1 = tf.cast(index_vector1,dtype=np.float64)
        a = np.array([0.]*self.nb_separated+[1.]*(output.shape[1]-self.nb_separated))
        index_vector1 = tf.matmul(index_vector1[:,None],a[None,:])
        index_vector1 = tf.cast(index_vector1!=0,dtype=tf.float64)
        #-- Part 2 of the network
        index_vector2 = tf.reduce_any(inputs[:,self.size_input:]!=0,axis=1)
        index_vector2 = tf.cast(index_vector2,dtype=tf.float64)
        a = np.array([1.]*self.nb_separated+[0.]*(output.shape[1]-self.nb_separated))
        index_vector2 = tf.matmul(index_vector2[:,None],a[None,:])
        index_vector2 = tf.cast(index_vector2!=0,dtype=tf.float64)

        #Combine both parts
        index_vector= index_vector1 + index_vector2

        #Apply noise depending on previously computed mask
        noisy_output = output+noise_vector*index_vector
        
        return noisy_output
