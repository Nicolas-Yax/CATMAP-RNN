import tensorflow as tf
import os
import re
import numpy as np
from scipy.io import savemat

def model_to_dict(model):
    """ Returns a dictionary compatible with matlab containing data from the given model """
    d = {}
    layers = []
    for l in model.layers:
        layer = {}
        layer["name"] = l.name
        layer["weights"] = {}
        for w in l.weights:
            name = re.findall('.*/([a-zA-Z0-9:_]*)',w.name)[-1]
            layer["weights"][name] = w.numpy()
        layers.append(layer)
        try:
            layer['activation'] = l.activation.__name__
        except:
            try:
                layer['activation'] = l.cell.activation.__name__
            except:
                pass
        try:
            layer['compute_mask'] = l.cell.mask
        except:
            pass
    d["layers"] = layers
    return d

def save_mat_name(name):
    """ Save the tf saved model of given name in a .mat file with the same name """
    model = tf.keras.models.load_model(os.path.join("models",name+".m"))
    save_mat_model(model,name)

def save_mat_model(model,name):
    """ Save the given model in a .mat file of given name """
    d = model_to_dict(model)
    print_dict(d)
    savemat(os.path.join("mat/",name+".mat"),d)

def save_mat_agent(agent,name):
    """ Save the given model in a .mat file of given name"""
    agent.load(name)
    model = agent.nn
    save_mat_model(model,name+"-"+agent.postname)
    
def print_dict(d,lvl=0):
    """ Print the given dictionary (lvl argument is used for recursion and shouldn't be changed) """
    if isinstance(d,dict):
        #print(" "*lvl+"{")
        for k in d.keys():
            print(" "*lvl+k+":")
            print_dict(d[k],lvl=lvl+1)
        #print(" "*lvl+"}")
    if isinstance(d,list):
        print(" "*lvl+"[")
        for i in range(len(d)):
            print_dict(d[i],lvl=lvl+1)
            print(" "*lvl+",")
        print(" "*lvl+"]")
    if isinstance(d,np.ndarray):
        print(" "*lvl+str(d.shape))
    if isinstance(d,str):
        print(" "*lvl+d)

from agents.control_rnn import *
from agents.pre_rnn import *
from agents.post_rnn import *
        
if __name__ == '__main__':
    name = 'std-category-noisy05-2'
    sp = (10,2)
    control = ControlRNN(50)
    save_mat_agent(control,name)
    pre = PreRNN(50)
    save_mat_agent(pre,name)
    post = PostRNN(50)
    save_mat_agent(post,name)
