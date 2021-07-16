from agents.llh_agent import *
import numpy as np

class ObservationAgent(LLHAgent):
    """ Agent computing LLH (computes according to PRE mapping) """
    def predict(self,batch):
        #Angles observed
        obs = batch.get('obs')[:,:,:2]
        complex_obs = obs[:,:,0]+obs[:,:,1]*1j
        angles_obs = np.angle(complex_obs)
        #Refs observed
        refs = batch.get('obs')[:,0,2:]
        complex_refs = refs[:,0]+refs[:,1]*1j
        angle_refs = np.angle(complex_refs)
        #Llh computation
        probas0,probas1 = self.llh_from_angles(angles_obs,angle_refs[:,None])
        return probas1>probas0
