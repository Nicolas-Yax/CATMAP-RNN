from env import *
from agent import *
from nn import *
from bayes_agent import *

nb_cards = 12

nn = simple_net((nb_cards,4),50,2)

env = Env(nb_cards)
batch,labels = env.generate_batch(2000)

agent = Agent(nn)
print("eval before",agent.evaluate(batch,labels))

#Training
for _ in range(500):
    batch,labels = env.generate_batch(100)
    agent.fit(batch,labels)

#Evaluation
batch,labels = env.generate_batch(2000)
print("eval after",agent.evaluate(batch,labels))

print("bayes eval",BayesAgent().evaluate(batch,labels))

