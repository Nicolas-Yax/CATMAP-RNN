from agent import Agent
import numpy as np

class LLHAgent(Agent):
    """ Abstract class for agents using llh """
    def __init__(self,kappa=0.5):
        self.kappa = kappa
    def llh(self,a,r):
        """ llh for orange angle seeing angle a with reference r """
        return np.exp(self.kappa*np.cos(a-r))
    def llh_from_angles(self,angles,refs):
        """ returns (orange_llh,blue_llh) of the full batch """
        probas0batch = self.llh(angles,refs)
        probas1batch = self.llh(angles,(refs+np.pi))
        probas0 = np.prod(probas0batch,axis=1)
        probas1 = np.prod(probas1batch,axis=1)
        return probas0,probas1
