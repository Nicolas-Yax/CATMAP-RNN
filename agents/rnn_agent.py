import tensorflow as tf
from tensorflow.keras.layers import *
from agent import Agent
import numpy as np

import matplotlib.pyplot as plt
import os

class RNNAgent(Agent):
    def __init__(self,input_shape,hidden_shape,out_shape,lr=10**-3):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.out_shape = out_shape
        self.lr = lr

        self.reset()

        self.layer_list = []

    def get_path_weights(self,mname,wname):
        path = os.path.join('models',mname+"-"+self.postname,wname)
        os.makedirs(path, exist_ok=True)
        return path
    
    def save(self,sname): #save name
        print("SAVING",self.layer_list)
        #self._save(sname,self.layer_list)
        for name in self.layer_list:
            getattr(self,name).save_weights(self.get_path_weights(sname,name))

    def load(self,sname):
        print("LOADING",self.layer_list)
        #self._load(sname,self.layer_list)
        for name in self.layer_list:
            getattr(self,name).load_weights(self.get_path_weights(sname,name))

    def _save(self,sname,lname):
        for name in lname:
            np.save(self.get_path_weights(sname,name),getattr(self,name).get_weights())

    def _load(self,sname,lname):
        for name in lname:
            w = np.load(self.get_path_weights(sname,name)+'.npy',allow_pickle=True)
            getattr(self,name).set_weights(w)
        
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

    def predict(self,batch,return_probas=False,store_probas=False):
        probas_out = self.forward(batch)
        actions = self.actions_from_probas(probas_out)
        if store_probas:
            #select probas of actions
            indices = [[i,actions[i]] for i in range(actions.shape[0])]
            probas = tf.gather_nd(probas_out,indices)
            batch.set('old_probas',probas)
            batch.set('old_actions',actions)
        if return_probas:
            return actions,probas_out
        return actions

    def evaluate(self,batch):
        actions = self.predict(batch)
        acc_array = actions.numpy()==batch.get('color')
        acc = np.mean(acc_array)
        return acc
    
    def reinforce_loss(self,batch):
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
    
    def ppo_loss(self, batch,clip_epsilon=0.1,i=None):
        """ Computes the PPO loss from a batch and labels """
        # Get probas for actions
        out_actions, out_probas = self.predict(batch, return_probas=True, store_probas=False)#(i==0))
        # Tensorflow way to get proba associated with chosen actions
        indices = [[i, out_actions[i]] for i in range(out_actions.shape[0])]
        pact = tf.gather_nd(out_probas, indices)
        # Rewards computation
        rews = tf.cast(tf.equal(out_actions, batch.get('color')), dtype=tf.float64) * 2 - 1
        #normalize rewards
        rews = rews - tf.reduce_mean(rews)
        # Old probas for actions
        old_actions, old_probas = batch.get('old_actions'), batch.get('old_probas')
        # Compute the ratio of new probas to old probas
        ratio = tf.exp(tf.math.log(pact) - tf.math.log(old_probas))
        # Compute the surrogate loss
        surrogate_loss = tf.minimum(ratio * rews, tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * rews)
        # Compute the clipped loss
        clipped_loss = -tf.reduce_mean(surrogate_loss)
        return clipped_loss

    def supervised_crossentropy_loss(self,batch):
        lbl = batch.get('color')
        out = self.forward(batch)
        loss = tf.keras.losses.sparse_categorical_crossentropy(lbl,out)
        lmean = tf.reduce_mean(loss)
        return loss

    def loss(self,batch,i=None):
        #return self.ppo_loss(batch,i=i)
        return self.reinforce_loss(batch)

    def parameters(self):
        return self.nn.weights
    
    def fit(self,batch,nb_fit=5):
        """ Fit once the model and given batch and labels """
        for i in range(nb_fit):
            self.opt.minimize(lambda : self.loss(batch,i=i),var_list=self.parameters())

    def train(self,env,nb,batch_size=2000,nb_fit=5,verbose=1):
        lscores = []
        for i in range(nb):
            batch = env.sample_batch(batch_size)
            eval_score = self.evaluate(batch)
            self.scores.append(eval_score)
            lscores.append(eval_score)
            if len(lscores)>10:
                del lscores[0]
            if verbose>=1 or verbose >=2:
                print(i,np.mean(lscores))
            self.fit(batch,nb_fit)

    def plot_scores(self,alpha=0.5,color='blue'):
        plt.plot([i for i in range(len(self.scores))],self.scores,alpha=alpha,color=color)
