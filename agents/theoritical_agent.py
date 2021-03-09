from agents.llh_agent import LLHAgent
import numpy as np

class TheoriticalAgent(LLHAgent):
    """ Agent predicting colors from the mean value used to sample angles (should be 100% accuracy as this mean value should always be centered on a color - it's a debug agent) """
    def predict(self,batch):
        refs = batch.get('ref')
        means = batch.get('mean')
        probas0,probas1 = self.llh_from_angles(means,refs[:,None])
        return probas1>probas0
