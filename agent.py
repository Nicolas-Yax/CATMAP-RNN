import tensorflow as tf
import numpy as np
class Agent:
    """ RNN Agent """
    def __init__(self,nn):
        #Model
        self.nn = nn
        #Optimizer
        self.opt = tf.keras.optimizers.Adam()

    def sample_outputs(self,out):
        """ Samples actions from a batch of probabilities """
        return tf.random.categorical(out,1)[:,0]

    def evaluate(self,batch,label):
        """ Computes accuracies from a batch and labels """
        out = self.nn(batch)
        return np.mean((out.numpy()>0.5)[:,1] == label)

    def loss(self,batch,label):
        """ Computes the loss from a batch and labels """
        #Get probas for colors
        out_probas = self.nn(batch)
        #Sample colors from probas
        out_actions = self.sample_outputs(out_probas)
        #Tensorflow way to get proba associated with chosen colors
        indices = [[i,out_actions[i].numpy()] for i in range(out_actions.shape[0])]
        pact = tf.gather_nd(out_probas,indices)
        #Rewards computation
        rews = (out_actions.numpy()==label)*2-1
        #Loss computation
        l = rews*tf.math.log(pact)
        lmean = tf.reduce_mean(l)
        return -lmean
    
    def fit(self,batch,label):
        """ Fit once the model and given batch and labels """
        self.opt.minimize(lambda : self.loss(batch,label),self.nn.weights)
