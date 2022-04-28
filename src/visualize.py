from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import numpy as np
import os
from abc import ABC, abstractmethod
import torch
import seaborn as sns
from src.utils.object import Object

class Plot(ABC):

    @abstractmethod
    def make_figure(self):
        pass

    def show_figure(self):
        self.make_figure()
        plt.show()
    
    def save_figure(self, name, path='./plots'):
        os.makedirs(path, exist_ok=True)
        self.make_figure()
        plt.savefig(f'{path}/{name}.pdf')
        plt.savefig(f'{path}/{name}.png')


class TetwisePlot(Plot):

    def __init__(self, object: Object, tet_colors: np.ndarray = None):
        self.object = object
        if tet_colors is None:
            self.tet_colors = np.array([object.vertices[t].mean() for t in object.tetrahedra])
        else:
            self.tet_colors = tet_colors

    def make_figure(self):
        # plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # colormap = cm.get_cmap('vlag_r')
        # colormap = cm.get_cmap('bwr_r')
        # colormap = cm.get_cmap('seismic_r')
        colormap = cm.get_cmap('coolwarm_r')

        # colormap = sns.color_palette("vlag", as_cmap=True)
        tetcolors = Normalize(self.tet_colors.min(),self.tet_colors.max())(self.tet_colors)
        tetcolors = colormap(tetcolors)
        ax.add_collection3d(Poly3DCollection([self.object.vertices[t] for t in self.object.tetrahedra], facecolors=tetcolors))
        axlim = 6.5
        ax.set_xlim([-axlim, axlim])
        ax.set_ylim([-axlim, axlim])
        ax.set_zlim([-axlim, axlim])
        ax.view_init(120, -90)

        m = cm.ScalarMappable(cmap=colormap)
        zmin = np.array([self.object.vertices[t].mean() for t in self.object.tetrahedra]).min()
        zmax = np.array([self.object.vertices[t].mean() for t in self.object.tetrahedra]).max()
        m.set_array([zmin, zmax])
        m.set_clim(zmin, zmax)
        fig.colorbar(m)
        ax.set_axis_off()

class LossPlot(Plot):

    def __init__(self, txt_file):
        loss_array = np.loadtxt(txt_file, delimiter=',', skiprows=1)
        self.bend_loss, self.twist_loss, self.joint_loss = loss_array.T

    def make_figure(self):
        fig = plt.figure()
        # plt.plot(self.bend_loss, label='Bend')
        # plt.plot(self.twist_loss, label='Twist')
        plt.plot(self.joint_loss, label='Total')
        plt.legend()
        


if __name__ == '__main__':

    obj = Object('./data/long_hammer/long_hammer_f1494_v749.obj',
                './data/long_hammer/long_hammer_f1494_v749.tet')
    params = torch.load('result_lr_1e3/parameter_100.pt')[1:].reshape(-1,3).detach().cpu().numpy()
    params = torch.load('max_param_bend_tetwise.pt')[1:].reshape(-1,3).detach().cpu().numpy()
    young_mod = params[:,0]*2.98
    TetwisePlot(obj, tet_colors=young_mod).save_figure('cmap')
    # LossPlot('result_lr_1e3_old/loss.txt').save_figure('total')