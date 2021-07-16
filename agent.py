import numpy as np

class Agent:
    """ Abstract class for agents """
    def predict(self,batch):
        raise NotImplementedError
    def evaluate(self,batch):
        predicted_labels = self.predict(batch)
        true_labels = batch.get('color')
        acc_array = predicted_labels==true_labels
        acc = np.mean(acc_array)
        return acc
