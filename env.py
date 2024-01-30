import numpy as np
from batch import Batch
from agents.angle_agent import *

class Env:
    """ CATMAP Environment class """
    def __init__(self,deck_size,kappa=0.5,optimal=False,condition='control'):
        """ deck_size is the length of sequences, kappa is the vonmises parameter and optimal=True returns optimal labels instead of real ones (accuracy of 1 reachable) """
        self.deck_size = deck_size
        self.kappa = kappa
        self.ref_angles = [2*np.pi*i/8 for i in range(8)]
        self.optimal_labels = optimal
        self.condition = condition

    def reset(self):
        """ Sample new generative parameters (new ref and new color) """
        #Sample new generative parameters
        self.ref = np.random.choice(self.ref_angles,1)[0]
        self.color = np.random.randint(0,2) #[0,1]

    def sample_batch(self,size):
        """ Samples a Batch of specified size from CATMAP Environment """
        batch = Batch()
        for i in range(size):
            #Get new generative parameters
            self.reset()
            #Add to batch
            batch.add('ref',self.ref)
            batch.add('color',self.color)
            if self.condition == 'pre':
                obs = [0,0,np.cos(self.ref),np.sin(self.ref)]
                batch.add('obs',obs)
            #Sample the sequence
            for j in range(self.deck_size):
                mean = (self.ref+self.color*np.pi)
                angle = np.random.vonmises(mean,self.kappa)
                #angle /= 2
                if self.condition == 'control':
                    obs = [np.cos(angle),np.sin(angle),np.cos(self.ref),np.sin(self.ref)]
                else:
                    obs = [np.cos(angle),np.sin(angle),0,0]
                #Add to batch
                batch.add('mean',mean)
                #batch.add('eps',eps)
                batch.add('angle',angle)
                batch.add('obs',obs)
            if self.condition == 'post':
                obs = [0,0,np.cos(self.ref),np.sin(self.ref)]
                batch.add('obs',obs)
        #Reshape the batch
        batch.numpy()
        batch.reshape('mean',(size,self.deck_size))
        #batch.reshape('eps',(size,self.deck_size))
        batch.reshape('angle',(size,self.deck_size))

        obs_size = self.deck_size if self.condition == 'control' else self.deck_size+1
        batch.reshape('obs',(size,obs_size,4))
        #Put optimal labels
        if self.optimal_labels:
            lbls = AngleAgent().predict(batch)
            batch.set('color',lbls)
        return batch
