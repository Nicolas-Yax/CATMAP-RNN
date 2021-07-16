from agents.theoritical_agent import * #Optimal agent
from agents.angle_agent import * #Optimal agent
from agents.observation_agent import * #Optimal agent
from agents.control_rnn import * #Control netorks
from agents.pre_rnn import * #Pre networks
from agents.post_rnn import * #Post networks
from env import * #Env

#Constant parameters
kappa = 0.5
deck_size = 12

#Creation of environment
env = Env(deck_size,kappa=kappa)

#Agent
nb_ep = 1000

#Training
agent = ControlRNN(50,lr=10**-3,random_training=True)
agent.train(env,nb_ep,verbose=1)

#Save
agent.save('test_agent')

#Load
agent2 = ControlRNN(50,lr=10**-3,random_training=False)
agent2.load('test_agent')

#Accuracy check
batch = env.sample_batch(2000) #batch is a dictionary that contains many things such as the input batch, labels and data about how this batch has been generated (all these data aren't used by the agent)
acc = agent.evaluate(env.sample_batch(2000))
print('accuracy',acc)

#Parameters for networks
simple_agent = ControlRNN(50,lr=10**-3,noise=0.5,random_training=True,activation='tanh')
"""
- lr is the learning rate used to decrease the loss
- noise is the stddev for noisy computation
- random_training=True means it will stop the computation after having seen a random number of angles of the sequence (between 1 and 12) making it learn on sequences of various length
- activation is the activation function used in the hidden layer
"""
separated_agent = SeparedControlRNN(50,separed_params=(10,2),free_feedbacks=True)
"""
- separed_params is a tuple (nb_units,nb_input) representing the number of units added to the layer which will see the first nb_input part of the input. Using (10,2) means it wil add 10 neurons which will the first 2 components of the input vector while the 50 standard neurons will see part of the input strictly after the first 2 components
- free feedback=False means SEQ -> REF connections will be cut. Setting it to True keeps the recurrence matrix full
"""
