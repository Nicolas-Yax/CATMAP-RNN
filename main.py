from agents.theoritical_agent import *
from agents.angle_agent import *
from agents.observation_agent import *
from agents.control_rnn import *
from agents.pre_rnn import *
from agents.post_rnn import *
from env import *
import numpy as np

import matplotlib.pyplot as plt

kappa = 0.5

env = Env(12,kappa=kappa)
agent = PostRNN(50,lr=10**-3)
agent.train(env,10000,batch_size=2000,nb_fit=1)
agent.plot_scores(color='green')
"""
for i in range(3):
    agent = ControlRNN(50,lr=10**-3)
    agent.train(env,2000,batch_size=2000,nb_fit=1)
    agent.plot_scores(color='grey')
    
    agent = PreRNN(50,lr=10**-3)
    agent.train(env,2000,batch_size=2000,nb_fit=1)
    agent.plot_scores(color='blue')
    
    agent = PostRNN(50,lr=10**-3)
    agent.train(env,2000,batch_size=2000,nb_fit=1)
    agent.plot_scores(color='green')
"""
plt.show()

