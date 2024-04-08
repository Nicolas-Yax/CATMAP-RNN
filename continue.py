
import argparse

from agents.theoritical_agent import *
from agents.angle_agent import *
from agents.observation_agent import *
from agents.control_rnn import *
from agents.pre_rnn import *
from agents.post_rnn import *
from agents.universal_rnn import *

from env import *
import numpy as np
from tf_to_mat import *


tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('-agent', type=str, default='universal')
parser.add_argument('-save_path', type=str,)
parser.add_argument('-rnn_noise', type=float, default=0.0)
parser.add_argument('-kappa', type=float, default=0.5)
parser.add_argument('-condition', type=str, default='pre')
parser.add_argument('-deck_size', type=int, default=12)
parser.add_argument('-nb_episodes', type=int, default=3000)
parser.add_argument('-batch_size', type=int, default=2000)
parser.add_argument('-nb_fit', type=int, default=5)
parser.add_argument('-name', type=str, default='0')

args = parser.parse_args()

if __name__ == '__main__':
    agent_ = args.agent
    rnn_noise = args.rnn_noise
    kappa = args.kappa
    condition = args.condition
    deck_size = args.deck_size
    nb_episodes = args.nb_episodes
    batch_size = args.batch_size
    nb_fit = args.nb_fit
    name = args.name

    env = Env(deck_size,kappa=kappa,condition=condition)

    if agent_ == 'universal':
        agent = UniversalRNN(64, activation='tanh', noise=rnn_noise)
    elif agent_ == 'lazy':
        agent = LazyUniversalRNN(64, activation='tanh', noise=rnn_noise)

    agent.postname = condition
    agent.load(args.save_path)
    agent.train(env,nb_episodes,batch_size=batch_size,nb_fit=nb_fit,verbose=1)
    agent.save(agent_+'-'+str(condition)+'-'+str(rnn_noise)+'-'+name)
    np.save('models/'+agent_+'-'+str(condition)+'-'+str(rnn_noise)+'-'+name+'/scores.npy',np.array(agent.scores))
