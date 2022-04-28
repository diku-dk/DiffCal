from abc import ABC, abstractproperty, abstractmethod
import torch
import numpy as np
from cached_property import cached_property

from src.camera import StaticCamera, Camera
from src.render import StaticRender, Render
from src.parameters import Parameters, TetwiseParameters
from src.descriptor import Descriptor

class Loss(ABC):

    def __init__(self, descriptor: Descriptor, parameters: Parameters, render: Render, camera: Camera):
        self.descriptor = descriptor
        self.camera = camera
        self.render = render
        self.parameters = parameters

    @abstractproperty
    def value(self):
        pass

    @abstractproperty
    def grad(self):
        pass



class L2Loss(Loss):

    @property
    def loss(self):
        error =  self.camera.target_images - self.render.rendered_images
        losses = torch.norm(error, p=2, dim=(1, 2, 3))
        loss_weights = torch.ones_like(losses)
        loss = torch.sum(losses * loss_weights)
        return loss
    
    @property
    def regularization_loss(self):
        return torch.tensor(0, device=self.descriptor.device, dtype=torch.float32)

    @property
    def value(self):
        # with open('twist_1e4_loss.txt', 'a') as f:
        #     print(1e3 * self.loss + 1e-1 * self.regularization_loss, file=f)
        return 1e3 * self.loss + 1e-1 * self.regularization_loss

    @property
    def grad(self):
        self.value.backward()
        grads = torch.zeros_like(self.parameters.parameter_tensor)
        should_optimize_density = bool(self.parameters.optimizable[0]) 
        grads[0] = 0.0 if not should_optimize_density else self.parameters.density_tensor.grad.data
        optimizable_material_params = self.parameters.optimizable[1:].to(self.descriptor.device)
        grads[1:] = (self.parameters.material_tensor.grad.data * optimizable_material_params).reshape(-1)
        return grads


class TetwiseL2Loss(L2Loss):

    @property
    def regularization_loss(self):
        # 2-ring regularization
        material_parameters_reg = self.parameters.material_tensor
        mean_two_ring_parameters = torch.zeros_like(material_parameters_reg[:, 0], dtype=torch.float32, device=self.descriptor.device)
        for ni, n in enumerate(self.descriptor.scene.object.two_ring_neighbourhood):
            mean_two_ring_parameters[ni] = material_parameters_reg[np.unique(np.delete(n, [i for i in range(len(n)) if n[i] == -1], None)), 0].mean()
        regularization_loss = torch.sum(torch.norm(mean_two_ring_parameters - material_parameters_reg[:, 0]) / self.parameters.NUM_OF_MATERIALS)
        return regularization_loss


class TwoModeLoss(Loss):

    def __init__(self, parameters, descriptor_1, render_1, camera_1, descriptor_2, render_2, camera_2, alpha=0.5):
        self.parameters = parameters
        self.descriptor_1, self.render_1, self.camera_1 = descriptor_1, render_1, camera_1
        self.descriptor_2, self.render_2, self.camera_2 = descriptor_2, render_2, camera_2
        self.alpha = alpha
        self.alpha = 0
        self.loss_0 = self.get_loss_0()
        self.loss_1 = self.get_loss_1()

        with open('result_lr_1e4/loss.txt', 'a') as f:
            print('bend, twist, total', file=f)

    @abstractmethod
    def get_loss_0(self) -> Loss:
        pass
    
    @abstractmethod
    def get_loss_1(self) -> Loss:
        pass

    @property
    def value(self):
        value_0, value_1 = self.loss_0.value, self.loss_1.value
        joint_loss = self.alpha*value_0 + (1-self.alpha)*value_1 
        with open('result_lr_1e4/loss.txt', 'a') as f:
            print(f'{value_0}, {value_1}, {joint_loss}', file=f)

        print(value_0, value_1)
        return joint_loss

    @property
    def grad(self):
        grad_0, grad_1 = self.loss_0.grad, self.loss_1.grad


        normal_grad_0 = grad_0/torch.norm(grad_0, p=2)
        normal_grad_1 = grad_1/torch.norm(grad_1, p=2)

        print('Norm', torch.norm(grad_0, p=2), torch.norm(grad_1, p=2))
        print('Unit', torch.norm(normal_grad_0, p=2), torch.norm(normal_grad_1, p=2))
        print('DOT grad', torch.dot(grad_0, grad_1))
        print('DOT Unit', torch.dot(normal_grad_0, normal_grad_1))

        return self.alpha*grad_0 + (1-self.alpha)*grad_1


class TwoModeL2Loss(TwoModeLoss):

    def get_loss_0(self):
        return L2Loss(self.descriptor_1, self.parameters, self.render_1, self.camera_1)

    def get_loss_1(self):
        return L2Loss(self.descriptor_2, self.parameters, self.render_2, self.camera_2)

class TwoModeTetwiseL2Loss(TwoModeLoss):

    def get_loss_0(self):
        return TetwiseL2Loss(self.descriptor_1, self.parameters, self.render_1, self.camera_1)

    def get_loss_1(self):
        return TetwiseL2Loss(self.descriptor_2, self.parameters, self.render_2, self.camera_2)

    
        


if __name__ == '__main__':
    from src.simulation import StaticBendSimulation, StaticTwistSimulation
    path = './experiments/paper_experiments/lh_m+e_2_mode'
    desc_1 = Descriptor(f'{path}/lh_m+e_elasticity_tetwise.exp', device='cuda', set_seed=True)
    desc_2 = Descriptor(f'{path}/twist_e+m_elasticity_tetwise.exp', device='cuda', set_seed=True)
    bg_thr_1, bg_thr_2 = 0.4925, 0.523
    camera_1, camera_2 = StaticCamera(desc_1, bg_thr_1), StaticCamera(desc_2, bg_thr_2)
    init_params = [5e4, 2.5e6, 5.0]
    
    params = TetwiseParameters(desc_1, initial_parameters=init_params, optimizable=[0,1,0,0], no_perturb=True)
    sim_1, sim_2 = StaticBendSimulation(desc_1, params), StaticTwistSimulation(desc_2, params)
    render_1, render_2 = StaticRender(desc_1, sim_1, camera_1), StaticRender(desc_2, sim_2, camera_2)
    loss = TwoModeTetwiseL2Loss(params, desc_1, render_1, camera_1, desc_2, render_2, camera_2)
    print(loss.value)
    print(loss.grad)
