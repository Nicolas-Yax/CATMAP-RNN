import tensorflow as tf
from tensorflow.keras.layers import *
from agent import Agent
import numpy as np

import matplotlib.pyplot as plt

class RNNAgent(Agent):
    def __init__(self,input_shape,hidden_shape,out_shape,lr=10**-3):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.out_shape = out_shape
        self.lr = lr

        self.reset()

    def reset_rnn(self):
        raise NotImplementedError

    def reset_opt(self):
        self.opt = tf.keras.optimizers.Adam(self.lr)

    def reset_scores(self):
        self.scores = []

    def reset(self):
        self.reset_rnn()
        self.reset_opt()
        self.reset_scores()

    def forward(self,inn):
        raise NotImplementedError

    def actions_from_probas(self,probas):
        return tf.random.categorical(tf.math.log(probas),1)[:,0]

    def predict(self,batch,return_probas=False):
        probas_out = self.forward(batch)
        actions = self.actions_from_probas(probas_out)
        if return_probas:
            return actions,probas_out
        return actions

    def evaluate(self,batch):
        actions = self.predict(batch)
        acc_array = actions.numpy()==batch.get('color')
        acc = np.mean(acc_array)
        return acc
    
    def loss(self,batch):
        """ Computes the loss from a batch and labels """
        #Get probas for colors
        out_actions,out_probas = self.predict(batch,return_probas=True)
        #Tensorflow way to get proba associated with chosen colors
        indices = [[i,out_actions[i].numpy()] for i in range(out_actions.shape[0])]
        pact = tf.gather_nd(out_probas,indices)
        #Rewards computation
        rews = (out_actions.numpy()==batch.get('color'))*2-1
        #print("---",out_probas[:5],out_actions[:5],indices[:5],pact[:5],label[:5],rews[:5])
        #Loss computation
        l = rews*tf.math.log(pact)
        #print('--',l[:5])
        lmean = tf.reduce_mean(l)
        return -lmean

    def parameters(self):
        return self.nn.weights
    
    def fit(self,batch,nb_fit=5):
        """ Fit once the model and given batch and labels """
        for _ in range(nb_fit):
            self.opt.minimize(lambda : self.loss(batch),self.parameters())

    def train(self,env,nb,batch_size=2000,nb_fit=5):
        lscores = []
        for i in range(nb):
            batch = env.sample_batch(batch_size)
            eval_score = self.evaluate(batch)
            self.scores.append(eval_score)
            lscores.append(eval_score)
            if len(lscores)>100:
                del lscores[0]
            if i%100==0:
                print(i,np.mean(lscores))
            self.fit(batch,nb_fit)

    def plot_scores(self,alpha=0.5,color='blue'):
        plt.plot([i for i in range(len(self.scores))],self.scores,alpha=alpha,color=color)
