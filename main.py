from agents.theoritical_agent import *
from agents.angle_agent import *
from agents.observation_agent import *
from agents.control_rnn import *
from agents.pre_rnn import *
from agents.post_rnn import *
from env import *
import numpy as np
from tf_to_mat import *

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Use float64 because of crossentropy that could compute log(0) due to numerical approximation
tf.keras.backend.set_floatx('float64')

#Parameters
kappa = 0.5 #Kappa for the von mises
deck_size = 12 #length sequences
optimal_training = False #Training on optimal labels instead of real ones

nb = 1 #Nb of networks to train
batch_size = 2000 #Batch size for the training (length of the training)
noise = None #Std for noisy networks (put None if you don't want a noisy network)
random_training = True #Training on sequence of random length chosen uniformely in [|1,12|]
separated_params = (10,2) #(Nb of neurons which will see the first part of the input,size of the first part of the input)

free_feedback = True #True = free feedback in the recurrent network / False = only REF -> SEQ feedbacks

activation = 'tanh' #Activation function

separated = False #Separated network

if optimal_training:
    random_training = False #Random training cannot be used with random_training (technical issues)

scores = [[],[],[],[]]
#Main Loop
for _i in range(nb):
    #Init plot curves
    curves = np.zeros((3,batch_size))
    #Init Env
    env = Env(deck_size,kappa=kappa)
    #---
    if optimal_training:
        envs = [Env(i,kappa=kappa,optimal=True) for i in range(1,13)]
    # Creates separated networks
    if separated:
        agent1 = SeparatedControlRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation,separated_params=separated_params,free_feedback=free_feedback)
        agent2 = SeparatedPreRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation,separated_params=separated_params,free_feedback=free_feedback)
        agent3 = SeparatedPostRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation,separated_params=separated_params,free_feedback=free_feedback)
    else: #Creates standard networks
        agent1 = ControlRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation)
        agent2 = PreRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation)
        agent3 = PostRNN(50,lr=10**-3,noise=noise,random_training=random_training,activation=activation)
    #Agents to train
    agents = [agent1,agent2,agent3]
    #Name for the save file and plot
    name = 'std-category-'+str(_i+2)
    for i,agent in enumerate(agents):
        #---
        if optimal_training:
            for k in range(batch_size):
                ind = np.random.randint(0,12)
                env = envs[ind]
                agent.train(env,1,batch_size=2000,nb_fit=5,verbose=1)
        else:
            #Train the netork
            agent.train(env,batch_size,batch_size=2000,nb_fit=5,verbose=1)
        #Evaluates the network
        env = Env(deck_size,kappa=kappa)
        acc = agent.evaluate(env.sample_batch(20000))
        print(i,"acc :",acc)
        #Save accuracies
        curves[i] += np.array(agent.scores)
        #Save the network
        agent.save(name)
    #Plot accuracies
    plt.plot([i for i in range(batch_size)],curves[0],alpha=0.4,color='grey',label='control')
    plt.plot([i for i in range(batch_size)],curves[1],alpha=0.4,color='blue',label='pre')
    plt.plot([i for i in range(batch_size)],curves[2],alpha=0.4,color='green',label='post')
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()
