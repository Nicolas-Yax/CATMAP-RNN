import numpy as np
from viewer import *

DEBUG = False

class BayesAgent:
    """ 'Optimal' Agent """
    def __init__(self):
        pass

    def evaluate(self,batch,label):
        """ Evaluate the agent """
        #Get mean angle
        mean_complex = np.mean(batch[:,:,:2],axis=1)
        mean_angles = np.arctan(mean_complex[:,1]/mean_complex[:,0])
        #Get color from these mean angles
        colors = self.angle_to_color(mean_angles,np.arctan(batch[:,0,3]/batch[:,0,2]))
        #Plot if DEBUG is True
        if DEBUG:
            view = Viewer()
            view.plot_circle(label[0])
            view.plot_colors(batch[0,0,2])
            for i in range(batch.shape[1]):
                view.plot_point(batch[0,i,:2])
            color = 'red'
            if colors[0]:
                color = 'lightblue'
            view.plot_point(mean_complex[0],color=color)
            print("color : ",colors[0])
            view.show()
        return np.mean(colors==label)

    def angle_to_color(self,angle,ref):
        """ Returns the color associated with given angle and ref angle """
        ref = ref
        angle = angle
        orange_area1 = abs((angle - ref)%np.pi) <= np.pi/4
        orange_area2 = abs((angle - ref)%np.pi) > np.pi/4 + np.pi/2
        orange_area = np.logical_or(orange_area1,orange_area2)
        return np.logical_not(orange_area)
