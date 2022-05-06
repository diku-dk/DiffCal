from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, ListedColormap
import numpy as np
import os
from abc import ABC, abstractmethod
import torch
import seaborn

from src.utils.object import Object
from src.experiment_types import Experiment, StaticHangExp



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
        plt.savefig(f'{path}/{name}.pdf', bbox_inches='tight')
        plt.savefig(f'{path}/{name}.png', dpi=300, bbox_inches='tight')


class TetwisePlot(Plot):

    def __init__(self, object: Object, param_file: str):
        self.object = object
        self.lame_params = torch.load(param_file)[1:].reshape(-1,3)

    def cutoff_params(self):
        mat_min = torch.tensor((5e0, 1e4, 0.5), device='cuda', dtype=torch.float32)
        mat_max = torch.tensor((10e4, 1e8, 100.0), device='cuda', dtype=torch.float32)
        lame = torch.zeros_like(self.lame_params)
        for p_id, parameter in enumerate(self.lame_params):
            lame[np.arange(self.lame_params.shape[0]) == p_id] = torch.max(torch.min(mat_max, parameter), mat_min)
        return lame

    @staticmethod
    def custom_colormap():
        top = cm.get_cmap('Oranges_r', 128)
        bottom = cm.get_cmap('Blues', 128)
        own_colormap = []
        red = np.array([1.0, 0.1, 0.1])
        blue = np.array([0.1, 0.1, 1.0])
        gray = np.array([0.75, 0.75, 0.75])
        for i in range(256):
            if i < 128:
                gray_percent = i / 128.
                red_percent = (128 - i) / 128.
                own_colormap.append(np.append(red * red_percent + gray * gray_percent, [1.0]))
            else:
                blue_percent = (i - 128) / 128.
                gray_percent = (128 - (i - 128)) / 128.
                own_colormap.append(np.append(blue * blue_percent + gray * gray_percent, [1.0]))

        return ListedColormap(own_colormap, name='RedGrayBlue')

    @property
    def young_modulus(self):
        # with possion=0.24, young_mod = 2.98*lambda
        # divided by 1000 to show values in kPa
        return (self.cutoff_params()[:,0]*2.98/1000).detach().cpu().numpy()

    def to_triangles(self):
        """
        Transforms the tetrahedron mesh to a triangle mesh
        :return:      An array of face-triangles of each tetrahedron.
        """
        young_modulus = self.young_modulus
        T = self.object.tetrahedra
        F = np.zeros((len(T) * 4, 3), dtype=int)
        youngs_modulus = np.zeros((len(T) * 4), dtype=float)
        for i in range(len(T)):
            t = T[i]
            F[i * 4 + 0] = (t[1], t[0], t[2])
            F[i * 4 + 1] = (t[0], t[1], t[3])
            F[i * 4 + 2] = (t[1], t[2], t[3])
            F[i * 4 + 3] = (t[2], t[0], t[3])
            youngs_modulus[i * 4 + 0] = young_modulus[i]
            youngs_modulus[i * 4 + 1] = young_modulus[i]
            youngs_modulus[i * 4 + 2] = young_modulus[i]
            youngs_modulus[i * 4 + 3] = young_modulus[i]
        return F, youngs_modulus

    def make_figure(self):
        F, facecolors = self.to_triangles()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colormap = self.custom_colormap()
        zmin = facecolors.min()
        zmax= facecolors.max()
        m = cm.ScalarMappable(cmap=colormap)
        m.set_array([zmin, zmax])
        m.set_clim(zmin, zmax)
        facecolors = Normalize(facecolors.min(),facecolors.max())(facecolors)
        facecolors = colormap(facecolors)
        V = self.object.vertices
        ax.add_collection3d(Poly3DCollection([V[f] for f in F], facecolors=facecolors))
        axlim = (V.max() - V.min()) / 2.
        ax.set_xlim([-axlim, axlim])
        ax.set_ylim([-axlim, axlim])
        ax.set_zlim([-axlim, axlim])
        ax.view_init(120, -90)
        ax.set_axis_off()
        colorbar = fig.colorbar(m, shrink=0.5)
        colorbar.ax.set_ylabel('Young\'s modulus (kPa)', rotation=0, fontsize=10, labelpad=-25, y=1.15)
        ax.set_axis_off()

class MeanTetwisePlot(TetwisePlot):

    def __init__(self, object: Object, param_files: list):
        self.object = object
        num_files, num_tets = len(param_files), self.object.tetrahedra.shape[0]
        lame = torch.zeros((num_files, num_tets, 3), dtype=torch.float32, device='cuda')
        all_lame_params = [TetwisePlot(self.object, param_file).lame_params for param_file in param_files]
        for i, lame_params in enumerate(all_lame_params):
            lame[i] = lame_params
        self.lame_params = torch.mean(lame, axis=0)

class LossPlot(Plot):

    def __init__(self, txt_file):
        loss_array = np.loadtxt(txt_file, delimiter=',', skiprows=1)
        self.bend_loss, self.twist_loss, self.total_loss = loss_array.T/1000

    def make_figure(self):
        fig = plt.figure()
        plt.plot(self.bend_loss, label='Hanging', color='tab:orange')
        plt.plot(self.twist_loss, label='Twisting', color='tab:green')
        plt.plot(self.total_loss, label='Total', color='tab:blue')
        # plt.grid()
        plt.xlabel('# Iterations')
        plt.ylabel('Loss')
        plt.legend()

class MeanLossPlot(Plot):
    def __init__(self, txt_files: list):
        self.bend = np.array([[LossPlot(txt_file).bend_loss] for txt_file in txt_files]).squeeze()
        self.twist = np.array([[LossPlot(txt_file).twist_loss] for txt_file in txt_files]).squeeze()
        self.total = np.array([[LossPlot(txt_file).total_loss] for txt_file in txt_files]).squeeze()
    

    def make_figure(self):
        plt.figure()
        plt.style.use('seaborn-whitegrid')
        x_axis = np.arange(self.bend.shape[1])
        mean_bend, std_bend = np.mean(self.bend, axis=0), np.std(self.bend, axis=0)
        mean_twist, std_twist = np.mean(self.twist, axis=0), np.std(self.twist, axis=0)
        mean_total, std_total = np.mean(self.total, axis=0), np.std(self.total, axis=0)
        plt.plot(x_axis, mean_bend, color='tab:orange', label='Hanging')
        plt.fill_between(x_axis, mean_bend-std_bend, mean_bend+std_bend, color='tab:orange', alpha=0.2)

        plt.plot(x_axis, mean_twist, color='tab:green', label='Twisting')
        plt.fill_between(x_axis, mean_twist-std_twist, mean_twist+std_twist, color='tab:green', alpha=0.2)

        plt.plot(x_axis, mean_total, color='tab:blue', label='Total')
        plt.fill_between(x_axis, mean_total-std_total, mean_total+std_total, color='tab:blue', alpha=0.2)
        
        plt.legend(fontsize=15)
        plt.xlim([0, 50])
        plt.ylim([164,176])
        plt.xlabel('# Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)


class TetwiseHistogram(Plot):

    def __init__(self, parameter: torch.tensor):
        self.parameter = parameter

    def make_figure(self):
        plt.hist(self.parameter, bins=20)
        plt.xlabel("Young's modulus (kPa)")
        plt.show()

class MatchingPlot(Plot):

    def __init__(self, exp: Experiment, fix_parameter: list):
        self.exp = exp
        self.exp.parameters.material_tensor
        self.exp.parameters.parameter_tensor = torch.tensor(fix_parameter, dtype=torch.float32, device=self.exp.descriptor.device)
        

    def make_figure(self):
        # self.exp.camera.show_images()
        # self.exp.render.show_images()
        render_img = (self.exp.render.rendered_images.detach().cpu().numpy().squeeze()*255).astype('uint8')
        target_img = (self.exp.camera.target_images.detach().cpu().numpy().squeeze()*255).astype('uint8')
        print(np.amax(target_img))
        mask = (render_img>0)
        matching_img = 132*np.ones_like(render_img)
        matching_img[~mask] = target_img[~mask]
        plt.imshow(matching_img, cmap='nipy_spectral')
        plt.show() 


if __name__ == '__main__':
    
    obj = Object('./data/long_hammer/long_hammer_f1494_v749.obj',
                './data/long_hammer/long_hammer_f1494_v749.tet')
    
    param_files = []
    i = 49
    for r in range(3,13):
        param_files.append(f'results/rep_{r}/parameters/p_{i}.pt')
    MeanTetwisePlot(obj, param_files).save_figure('tet_dist')

    # TetwiseHistogram(young_mod).save_figure('hist')
    # LossPlot('results/rep_3/loss.txt').save_figure('loss')

    # txt_files = [f'results/rep_{r}/loss.txt' for r in np.arange(3,13)]
    # MeanLossPlot(txt_files).save_figure('joint_loss')

    # exp = StaticHangExp(exp_file=['./experiments/paper_experiments/lh_e/lh_e_elasticity.exp'],
    #                     initial_parameters=[5e4, 2.5e6, 5.],
    #                     optimizable=[0,1,0,0],
    #                     num_iters=1,
    #                     perturb_type='none',
    #                     lr=1e3,
    #                     background_threshold=[0.520],
    #                     )

    # factory = 27852
    # optimal = 21812
    # MatchingPlot(exp, fix_parameter=[1080, optimal, 2.5e6, 5.]).make_figure()
