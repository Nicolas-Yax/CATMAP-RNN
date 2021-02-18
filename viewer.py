import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

def deg_from_rad(rad):
    return rad/(2*np.pi)*360

def rad_from_deg(deg):
    return deg/360*2*np.pi

class Viewer:
    """ Visualization class """
    def __init__(self):
        self.fig,self.ax = plt.subplots()

    def plot_circle(self,color):
        """ Plot the contour of the plot """
        guessed_color='orange'
        if color:
            guessed_color='blue'
        self.ax.plot([np.cos(x) for x in np.linspace(0,2*np.pi,1000)],[np.sin(x) for x in np.linspace(0,2*np.pi,1000)],color=guessed_color,linewidth=3)
        
    def plot_colors(self,ref):
        """ Plot background colors from the ref angle """
        orange_w1 = ptch.Wedge((0,0),1,deg_from_rad(ref)-45,deg_from_rad(ref)+45,color='orange')
        orange_w2 = ptch.Wedge((0,0),1,deg_from_rad(ref)-45+180,deg_from_rad(ref)+45+180,color='orange')
        blue_w1 = ptch.Wedge((0,0),1,deg_from_rad(ref)-45+90,deg_from_rad(ref)+45+90,color='blue')
        blue_w2 = ptch.Wedge((0,0),1,deg_from_rad(ref)-45+270,deg_from_rad(ref)+45+270,color='blue')
        self.ax.add_artist(orange_w1)
        self.ax.add_artist(orange_w2)
        self.ax.add_artist(blue_w1)
        self.ax.add_artist(blue_w2)
        
    def plot_angle(self,obs,color='red'):
        """ Plot given angle (cos,sin)"""
        self.ax.plot([-obs[0],obs[0]],[-obs[1],obs[1]],color=color,linewidth=5)

    def plot_point(self,obs,color='black'):
        """ Plot given point (x,y)"""
        self.ax.plot([obs[0]],[obs[1]],marker='o',color=color)

    def show(self):
        """ Show the plot """
        plt.show()
