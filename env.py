import numpy as np


angles = [2*np.pi*i/8 for i in range(8)] #orange angle
class Env:
    """ CATMAP Environment class """
    def __init__(self,nb_sample):
        self.nb_sample = nb_sample #Nb cards
        self.kappa = 0.5 #Von mises parameter

    def reset(self):
        """ Samples a new color and a new ref angle """
        self.ref = np.random.choice(angles,1)[0] #Angle for orange
        self.color = np.random.randint(0,2) #0:orange 1:blue

    def sample_obs(self):
        """ Sample a new angle from color and ref angle. """
        mean = self.ref+self.color*np.pi/2
        angle = np.random.vonmises(0,self.kappa)+mean
        self.obs = [np.cos(angle),np.sin(angle),np.cos(self.ref),np.sin(self.ref)]

    def generate_batch(self,size):
        """ Creates a full batch of given size and labels (colors) """
        self.reset()
        batch = np.zeros((size,self.nb_sample,len(self.generate_obs()))) #Empty batch
        label = np.zeros(size)
        for i in range(size):
            self.reset()
            for j in range(self.nb_sample):
                batch[i,j] = self.generate_obs()
            label[i] = self.color
        return batch,label
            
    def generate_obs(self):
        """ Creates an observation from current color and ref """
        self.sample_obs()
        return self.obs

    def render(self,show=False):
        """ Plot last sampled observation as well as current color and ref angle """
        view = Viewer()
        view.plot_circle(self.color)
        view.plot_colors(self.ref)
        view.plot_angle(self.obs)
        if show:
            view.show()
        return view

