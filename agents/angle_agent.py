from agents.llh_agent import LLHAgent
import numpy as np

class AngleAgent(LLHAgent):
    """ Agent predicting colors from angles directly (instead of using cos and sin encoding of angles """
    def predict(self,batch):
        refs = batch.get('ref')
        angles = batch.get('angle')
        probas0,probas1 = self.llh_from_angles(angles,refs[:,None])
        return probas1>probas0
