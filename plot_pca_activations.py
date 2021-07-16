from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np
from agents.post_rnn import *
from agents.pre_rnn import *
from agents.control_rnn import *
from env import *
import tensorflow as tf
from sklearn.decomposition import PCA
from analyse.plot_pca import *
from mpl_toolkits import mplot3d

#Set floats to 64 bits encoding 
tf.keras.backend.set_floatx('float64')

#Parameters of the plot
typ = "std" #'std' or 'separated' for the type of the network asked

name = typ+"-category-1" #name of the network to load

rnn = 'post' #mapping of the network
activation = 'tanh' #activation function

sep_params = (10,2) #params for separated netorks

residue_ref = False #Remove the reference from the latent space
residue_timestep = False #Remove the timestep counter from the latent space

nb_comp = 5 #Nb components to ask from the PCA
pca_batch_size = 2000 #Batch size to fit the PCA on
deck_size = 12 #Size of sequences

pca_axes = [0,1] #Axes of the pca to plot (3 max)

#Nb of components to plot
plot_dim = len(pca_axes)

#Create Env
env= Env(deck_size)

#Create Agent
if rnn == 'control':
    if typ == 'separated':
        agent = SeparatedControlRNN(50,activation=activation,separated_params=sep_params)
    else:
        agent = ControlRNN(50,activation=activation)
elif rnn == 'pre':
    if typ == 'separated':
        agent = SeparatedPreRNN(50,activation=activation,separated_params=sep_params)
    else:
        agent = PreRNN(50,activation=activation)
elif rnn == 'post':
    if typ == 'separated':
        agent = SeparatedPostRNN(50,activation=activation,separated_params=sep_params)
    else:
        agent = PostRNN(50,activation=activation)

#Load Agent
agent.load(name)

#Print agent acurracy
print('agent accuracy : ',agent.evaluate(env.sample_batch(20000)))

#Pre compute batch
batch_obsf = env.sample_batch(pca_batch_size) 
batch_obs = batch_obsf.get('obs')
if rnn == 'post' and not(typ=='separated'):
    batch_obs = batch_obs[:,:,:2]

#Compute output of rnn part of agent
if typ == 'separated':
    if rnn == 'control':
        out = agent.rnn(batch_obs).numpy()
    elif rnn == 'pre':
        x = batch_obs.copy()
        refs = x[:,[0],2:]
        zeros = tf.zeros(tf.shape(refs))
        refs = np.concatenate([zeros,refs],axis=2)
        x[:,:,2:] = 0
        x = np.concatenate([refs,x],axis=1)
        out = agent.rnn(x).numpy()
        #out = out[:,[0],:]
    elif rnn == 'post':
        x = batch_obs
        refs = x[:,[0],2:]
        zeros = tf.zeros(tf.shape(refs))
        refs = np.concatenate([zeros,refs],axis=2)
        x[:,:,2:] = 0
        x = np.concatenate([x,refs],axis=1)
        print(x.shape)
        out = agent.rnn(x).numpy()
        out = out[:,:,:]
        print(out.shape)
elif rnn == 'control':
    out = agent.rnn(batch_obs).numpy()
elif rnn == 'pre':
    init_state = agent.context_nn(batch_obs[:,0,2:])
    out = agent.rnn(batch_obs[:,:,:2],initial_state=init_state).numpy()
elif rnn == 'post':
    zero_state = tf.zeros((batch_obs.shape[0],50),dtype=tf.float64)
    out = agent.rnn(batch_obs,initial_state=zero_state).numpy()

#Def residue function
def residue(X,Y):
    from sklearn.linear_model import LinearRegression
    Lg = LinearRegression()
    XX = np.reshape(X,(-1,X.shape[-1]))
    YY = Y
    Lg.fit(YY,XX)
    print("score",Lg.score(YY,XX))
    #assert False
    pred = Lg.predict(YY)
    pred = np.reshape(pred,X.shape)
    return X - pred

#Compute residue of the reference if asked (residue_ref==True) 
if residue_ref:
    Y = batch_obs[:,0,2:]
    ref = np.concatenate([Y[:,None] for _ in range(out.shape[1])],axis=1)
    ref = np.reshape(ref,(-1,2))
    print(ref.shape)
    out = residue(out,ref)

#Compute residue of the timestep if asked (residue_timestep==True) 
if residue_timestep:
    timesteps = [range(out.shape[1]) for _ in range(batch_obs.shape[0])]
    timesteps = np.reshape(timesteps,(-1,1))
    out = residue(out,timesteps)

#Pre compute recurrent output of the netork
nnb = 8
if typ == 'separated':
    nb_n = 60
else:
    nb_n = 50

if rnn == 'post' and typ == 'separated':
    out = out[:,:-1,:]
    print('os',out.shape)

#Fit PCA on the output asking for a lot of components (to see as many component variance as possible)
pca = fit_pca(out.reshape((-1,nb_n)),nb_components=nnb)
#Print variance of PCA
print(pca.explained_variance_)
#Plot variance of PCA
plt.plot(range(nnb),pca.explained_variance_,marker='o')
plt.show()

#Fit the PCA on the number of component asked
pca = fit_pca(out.reshape((-1,nb_n)),nb_components=nb_comp)
print("PCA fitted")
print("variance : ",pca.explained_variance_)

#Create figure for latent space plot
if plot_dim == 3:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')
    ax.set_zlabel('pca3')
else:
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('pca1')
    ax.set_ylabel('pca2')

#Compute output transformed by PCA
print(out.shape,typ,nb_n)
transformed_batch = pca.transform(np.reshape(out,(-1,nb_n)))[:,pca_axes]
print(out.shape,transformed_batch.shape)
pca_batch = np.reshape(transformed_batch,(pca_batch_size,out.shape[1],len(pca_axes)))

#Plot background function
def plot_bg(zorder=0):
    colors = [(i/deck_size,(1-i/deck_size),(i-6)**2/36) for i in range(pca_batch.shape[1])]
    print("bg",pca_batch.shape)
    for i in range(pca_batch.shape[1]):
        if plot_dim == 3:
            ax.scatter(pca_batch[:,i,0],pca_batch[:,i,1],pca_batch[:,i,2],color=colors[i],alpha=0.1,marker='.',zorder=zorder)
        if plot_dim == 2:
            ax.scatter(pca_batch[:,i,0],pca_batch[:,i,1],color=colors[i],alpha=0.1,marker='.',zorder=zorder)

#Returns a grid of the latent space (for categorisation field)
def get_batch_state(res=10,return_grid=False):
    axis_x = np.linspace(np.min(pca_batch[:,:,0]),np.max(pca_batch[:,:,0]),10)
    axis_y = np.linspace(np.min(pca_batch[:,:,1]),np.max(pca_batch[:,:,1]),10)
    if plot_dim ==3:
        axis_z =np.linspace(np.min(pca_batch[:,:,2]),np.max(pca_batch[:,:,2]),10)

    if plot_dim ==3:
        X,Y,Z = np.meshgrid(axis_x,axis_y,axis_z)
        M=np.concatenate([X[:,:,:,None],Y[:,:,:,None],Z[:,:,:,None]],axis=3)
        print(M.shape)
        Zr = np.reshape(M,(-1,3))
    else:
        X,Y = np.meshgrid(axis_x,axis_y)
        Z = np.concatenate([X[:,:,None],Y[:,:,None]],axis=2)
        Zr = np.reshape(Z,(-1,2))

    proj_mat = pca.components_[pca_axes]

    Zrp = np.dot(Zr,proj_mat)

    batch_state = Zrp

    out = tf.convert_to_tensor(batch_state,dtype=tf.float64)
    if return_grid:
        return out,Zr
    return out

#Plot categories in the latent space
def plot_categorisation_field(zorder=10):
    c = 0.1
    batch_state,grid = get_batch_state(return_grid=True)
    if rnn == 'control' or typ == 'separated':
        out_batch = agent.dense(batch_state)
    elif rnn == 'pre':
        out_batch = agent.out_dense(batch_state)
    for k in range(out_batch.shape[0]):
        ax.plot([grid[k][0]],[grid[k][1]],[grid[k][2]],color='orange',marker='o',alpha=out_batch[k,0]*c,zorder=zorder)
        ax.plot([grid[k][0]],[grid[k][1]],[grid[k][2]],color='blue',marker='o',alpha=out_batch[k,1]*c,zorder=zorder)

#Plot the latent space
if rnn == 'post':
    plot_bg()
    
if rnn == 'pre':
    plot_bg()
    plot_categorisation_field()

if rnn == 'control':
    plot_bg()
    plot_categorisation_field()

plt.show()
