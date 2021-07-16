import numpy as np
import pickle

class Batch:
    """ Dictionary class containing many usefull things for training / testing / debuging. It contains different vectors labelled by strings (ex : batch['input'] might contain a volume of input and batch['lbl'] could be a volume of labels).
    This class has to be used according to a 2 phases scheme :
    - Phase 1 : set/add things using the add method. Internal structures uses list to make it simpler to add new values in the batch (simpler than using np.concatenat which would malloc each time it is called).
    - Phase 2 : get what you want from the batch. Internal structures uses np.array to simplify the use of store data.
    It starts in phase 1 (lists) and to switch to phase 2 (np.array):
    - call the numpy method to transform lists in np array
    - call the reshape method to give the desired shape to np array
 """
    def __init__(self):
        self.d = {}
    def add(self,lbl,v):
        try:
            self.d[lbl].append(v)
        except KeyError:
            self.d[lbl] = [v]
    def get(self,lbl):
        return self.d[lbl]
    def set(self,lbl,v):
        self.d[lbl] = v
    def save(self,f):
        pickle.dump(self.d,open(f,'wb'))
    def load(self,f):
        self.d = pickle.load(open(f,'rb'))
    def numpy(self):
        for k in self.d.keys():
            self.d[k] = np.array(self.d[k])
    def reshape(self,lbl,shape):
        self.d[lbl] = np.reshape(self.d[lbl],shape)
    @property
    def shape(self):
        l = {}
        for k in self.d.keys():
            l[k] = self.d[k].shape
        return l
