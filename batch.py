import numpy as np
import pickle

class Batch:
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
