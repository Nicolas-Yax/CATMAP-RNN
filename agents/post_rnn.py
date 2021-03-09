import tensorflow as tf
from tensorflow.keras.layers import *
from agents.rnn_agent import RNNAgent
import numpy as np

class PostRNN(RNNAgent):
    def __init__(self,hidden_shape,input_shape=(None,2),out_shape=2,context_input_shape=(2,),lr=10**-3):
        self.context_input_shape = context_input_shape
        super().__init__(input_shape,hidden_shape,out_shape,lr=lr)

    def reset_rnn(self):
        #Classic network
        inp = Input(shape=self.input_shape)
        init_inp = Input(shape=self.hidden_shape)
        x = inp
        x = SimpleRNN(self.hidden_shape,activation='tanh')(x,initial_state=init_inp)
        m = tf.keras.Model(inputs=[inp,init_inp],outputs=x)
        self.nn = m
    
        #Context network
        inp_rec_context = Input(shape=self.context_input_shape)
        inp_context = Input(shape=self.hidden_shape)
        x_context = tf.concat([inp_rec_context,inp_context],axis=1)
        x_context = Dense(self.hidden_shape,activation='tanh')(x_context)
        m_context = tf.keras.Model(inputs=[inp_context,inp_rec_context],outputs=x_context)
        self.context_nn = m_context

        #Out network
        inp_out = Input(shape=self.hidden_shape)
        x = inp_out
        #x = Dense(self.hidden_shape,activation='tanh')(x)
        out = Dense(self.out_shape,activation='softmax')(x)
        m_out = tf.keras.Model(inputs=inp_out,outputs=out)
        self.out_nn = m_out

    def parameters(self):
        return [self.nn.weights,self.context_nn.weights,self.out_nn.weights]
    
    def forward(self,batch):
        batch_obs = batch.get('obs')[:,:,:2]
        ref = batch.get('obs')[:,0,2:]
        zero_state = np.zeros((batch_obs.shape[0],self.context_nn.output_shape[1]))
        acc = self.nn([batch_obs,zero_state])
        acc = self.context_nn([acc,ref])
        out = self.out_nn(acc)
        return out
