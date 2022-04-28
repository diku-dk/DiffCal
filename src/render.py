from DifferentiableRenderWrappers.DiffRayTracer import DiffRayTracer
from cached_property import cached_property
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractproperty

from src.descriptor import Descriptor
from src.simulation import Simulation, StaticBendSimulation, StaticTwistSimulation, DynamicBendSimulation
from src.parameters import Parameters, TetwiseParameters
from src.camera import Camera, DynamicCamera, StaticCamera


class Render(ABC):

    def __init__(self, descriptor: Descriptor, simulation: Simulation, camera: Camera):
        self.descriptor = descriptor
        self.simulation = simulation
        self.camera = camera

    @abstractproperty
    def rendered_images(self) -> torch.Tensor:
        pass

    def show_images(self):
        self._make_figure()
        plt.show()

    def _make_figure(self):
        for rendered_image in self.rendered_images:
            plt.figure()
            plt.imshow((rendered_image.detach().cpu().numpy()*255).astype('uint8').squeeze())



class StaticRender(Render):
    

    @property
    def rendered_images(self) -> torch.Tensor:
        images = torch.flipud(self._render(self.camera.view)[0]).to(self.descriptor.device)[None, :]
        return images
    
    @cached_property
    def _diff_render(self) -> DiffRayTracer:
        """ The differentiable renderer used in the scene.
        """
        render_height, render_width = self.camera.target_images.shape[1:3]
        model = self.simulation.get_model()
        _diff_render = DiffRayTracer(render_width, render_height, render_type=0, anti_aliasing_kernel=(16, 16), device=self.descriptor.device)
        _diff_render.add_model(model.state().q, model.tri_indices)
        return _diff_render

    @cached_property
    def _diff_intrinsics(self):
        return self._diff_render.redner_intrinsic_mat(
                            fx=self.descriptor.experiment.camera_intrinsics[0, 0],
                            fy=self.descriptor.experiment.camera_intrinsics[1, 1],
                            cx=self.descriptor.experiment.camera_intrinsics[0, 2],
                            cy=self.descriptor.experiment.camera_intrinsics[1, 2])

    def _render(self, view):
        camera_transform = self.camera.view_transforms[view]
        if len(self._diff_render.cameras) == 0:
            self._diff_render.add_camera(cam_to_world=camera_transform, intrinsic_mat=self._diff_intrinsics)
        else:
            self._diff_render.replace_camera(0, cam_to_world=camera_transform, intrinsic_mat=self._diff_intrinsics)
        self._diff_render.update_model(0, self.simulation.run().q)
        return self._diff_render.render()


class DynamicRender(StaticRender):

    @property
    def rendered_images(self):
        images = torch.zeros(len(self.camera.target_images), *self.camera.target_images[0].shape, device=self.descriptor.device, dtype=torch.float32)
        for di, sub_steps in enumerate(self.camera.data_based_sub_steps):
            self.simulation.sim_steps = sub_steps
            images[di] = torch.flipud(self._render(di)[0]).to(self.descriptor.device)
        return images





if __name__ == '__main__':
    desc = Descriptor('./experiments/paper_experiments/twist_e+m/twist_e+m_elasticity.exp')
    init_params = [6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0]
    param = TetwiseParameters(desc, initial_parameters=init_params, optimizable=[0,1,0,0], no_perturb=True)
    sim_static = StaticTwistSimulation(desc, param)
    camera = StaticCamera(desc)
    StaticRender(desc, sim_static, camera).show_images()