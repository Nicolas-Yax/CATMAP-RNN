import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def fit_pca(data,nb_components=3):
    pca = PCA(n_components=nb_components)
    pca.fit(data)
    return pca

def project_pca(data,pca):
    print(data.shape)
    return pca.transform(data)

def plot_pca(data,pca,color='blue',alpha=1,show=True):
    projections = project_pca(data,pca)
    plt.plot(projections[:,0],projections[:,1],color=color,alpha=alpha)
    if show:
        plt.show()

def get_reference_pca(env,agent,batch_size=2000,nb_comp=3):
    out_batch = agent.rnn(env.sample_batch(batch_size).get('obs'))
    pca = fit_pca(out_batch[:,-1,:],nb_components=nb_comp)
    return pca

def get_pca_batch(env,agent,pca,batch_size=2000,return_in_batch=False):
    in_batch = env.sample_batch(batch_size)
    out_batch = agent.rnn(in_batch.get('obs'))[:,-1,:]
    pca_batch = project_pca(out_batch,pca)
    #pca_batch = out_batch
    if return_in_batch:
        return pca_batch,in_batch
    return pca_batch
