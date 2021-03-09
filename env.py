import numpy as np
from batch import Batch
from numpy.random import default_rng

rng = default_rng()

class Env:
    def __init__(self,deck_size,kappa=0.5):
        self.deck_size = deck_size
        self.kappa = kappa
        self.ref_angles = [np.pi*i/8 for i in range(8)]

    def reset(self):
        #Sample new generative parameters
        self.ref = np.random.choice(self.ref_angles,1)[0]
        self.color = np.random.randint(0,2)

    def sample_batch(self,size):
        batch = Batch()
        for i in range(size):
            #Get new generative parameters
            self.reset()
            #Add to batch
            batch.add('ref',self.ref)
            batch.add('color',self.color)
            #Sample the sequence
            for j in range(self.deck_size):
                mean = (self.ref+self.color*np.pi/2)%np.pi
                angle = rng.vonmises(mean,self.kappa)
                obs = [np.cos(angle),np.sin(angle),np.cos(self.ref),np.sin(self.ref)]
                #Add to batch
                batch.add('mean',mean)
                #batch.add('eps',eps)
                batch.add('angle',angle)
                batch.add('obs',obs)
        #Reshape the batch
        batch.numpy()
        batch.reshape('mean',(size,self.deck_size))
        #batch.reshape('eps',(size,self.deck_size))
        batch.reshape('angle',(size,self.deck_size))
        batch.reshape('obs',(size,self.deck_size,4))
        return batch
