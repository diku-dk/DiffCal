import torch
from cached_property import cached_property
from typing import List
import numpy as np

from src.descriptor import Descriptor

class Parameters:
    
    NUM_OF_PARAMS = 3   # [mu, lambda, damping]

    def __init__(self,
             descriptor: Descriptor,
             initial_parameters: list,
             optimizable: List[bool],
             perturb_type: str = 'large',
             initial_density: float = 1080.0,
             parameters: list = None,
             density: float = None):
        self._check_input(initial_parameters, optimizable, perturb_type)
        if descriptor.seed: descriptor.set_seed(0)
        self.descriptor = descriptor
        self.optimizable = torch.tensor(optimizable, dtype=torch.bool, device=self.descriptor.device)
        self.initial_density = initial_density
        self.perturb_type = perturb_type
        self.initial_param_list = initial_parameters
        self.density = initial_density if density is None else density
        if parameters is None:
            self.parameters = self._perturbed_initial_parameters.to(self.descriptor.device)
        else:
            self.parameters = torch.tensor(parameters, dtype=torch.float32, device=self.descriptor.device).reshape(self.NUM_OF_MATERIALS, self.NUM_OF_PARAMS)

    @property
    def parameter_tensor(self):
        return torch.cat((self.density_tensor.reshape(-1), self.parameters.reshape(-1)))
    
    @parameter_tensor.setter
    def parameter_tensor(self, tensor):
        density, material = tensor[0], tensor[1:]
        self.density = float(density.item())
        self.parameters = material.reshape(self.NUM_OF_MATERIALS, self.NUM_OF_PARAMS)
        del self.material_tensor

    @cached_property
    def tetwise_param_dist(self):
<<<<<<< HEAD
=======
        for i, bbox  in enumerate(self.descriptor.experiment.object_parameter_bounding_boxes):
            self.descriptor.scene.object.add_parameter_division(bbox, i)
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
        return self.descriptor.scene.object.tet_wise_parameter_distribution

    def distribute_material_parameters(self):
        """ Assigns Lamé (mu and lambda) and damping parameters to each tetrahedra.
        """
        tetwise_param_dist = np.empty(0, dtype=np.int)
<<<<<<< HEAD
=======
        for dbc in self.descriptor.experiment.object_dirichlet_boundary_conditions:
            self.descriptor.scene.object.add_dirichlet_boundary_conditions(dbc)
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
        tetwise_param_dist.resize(self.descriptor.scene.object.tetrahedra.shape[0])
        tet_count = tetwise_param_dist.shape[0]
        mat_min = torch.tensor((5e0, 1e4, 0.5), device=self.descriptor.device, dtype=torch.float32)
        mat_max = torch.tensor((25e4, 1e8, 100.0), device=self.descriptor.device, dtype=torch.float32)
        mat_val = torch.zeros((tet_count, self.NUM_OF_PARAMS), device=self.descriptor.device, dtype=torch.float32).contiguous()
        for p_id, parameter in enumerate(self.material_tensor):
            mat_val[self.tetwise_param_dist == p_id] = torch.max(torch.min(mat_max, parameter), mat_min)
        return mat_val

    @property
    def density_tensor(self):
        return torch.tensor(self.density, device=self.descriptor.device, dtype=torch.float32, requires_grad=bool(self.optimizable[0]))
    
    @cached_property
    def material_tensor(self) -> torch.Tensor:
        material_parameters = []
        for m in range(self.NUM_OF_MATERIALS):
            material_m = []
            if self._should_only_optimize_mu:
                material_m.append(self.parameters[m, 0])
                material_m.append(self._lambda(self.parameters[m, 0].item()))
            elif self._should_only_optimize_lambda:
                material_m.append(self._mu(self.parameters[m, 1]).item())
                material_m.append(self.parameters[m, 1])
            else:
                material_m.append(self.parameters[m, 0])
                material_m.append(self.parameters[m, 1])
            material_m.append(self.parameters[m, 2])
            material_parameters.append(material_m)
        return torch.tensor(material_parameters, device=self.descriptor.device, dtype=torch.float32, requires_grad=True)

    @cached_property
    def NUM_OF_MATERIALS(self):
        return len(self.initial_param_list) // self.NUM_OF_PARAMS

    @cached_property
    def initial_parameters(self):
        return torch.tensor(self.initial_param_list, dtype=torch.float32, device=self.descriptor.device).reshape(self.NUM_OF_MATERIALS, self.NUM_OF_PARAMS)

    def _check_input(self, initial_parameters, optimizable, perturb_type):
        if len(initial_parameters) % self.NUM_OF_PARAMS != 0:
            raise ValueError(f"Length of parameters should be divisible by {self.NUM_OF_PARAMS}.")
        if len(optimizable) != (self.NUM_OF_PARAMS + 1):   # +1 since we might also optimize density.
            raise ValueError(f"Length of optimizable should be {self.NUM_OF_PARAMS + 1}.")
        allowed_perturbs = ['none', 'small', 'large']
        if perturb_type not in allowed_perturbs:
            raise ValueError(f'perturb should be one of {allowed_perturbs}.')
    
    @property
    def _perturbed_initial_parameters(self):
        return (self.optimizable[1:] * self._perturbation) * (self.initial_parameters) + self.initial_parameters

    @cached_property
    def _perturbation(self):
        none_perturbation = torch.tensor(0, device=self.descriptor.device)
        small_perturbation = torch.normal(15, 5, size=(1,1)) * 0.01 * (torch.randint(2, size=(1,1))*2-1)
        large_perturbation = torch.normal(40, 5, size=(1,1)) * 0.01 * (torch.randint(2, size=(1,1))*2-1)
        perturb_dict = {'none': none_perturbation, 'small': small_perturbation, 'large': large_perturbation}
        return perturb_dict[self.perturb_type].to(self.descriptor.device)

    @cached_property
    def _should_only_optimize_mu(self):
        return self.optimizable[1] and not self.optimizable[2]

    @cached_property
    def _should_only_optimize_lambda(self):
        return self.optimizable[2] and not self.optimizable[1]

    @staticmethod    
    def _mu(lambda_, nu=0.49):
        E = (lambda_/nu) * (1+nu) * (1-2*nu)
        return E / (2*(1 + nu))

    @staticmethod
    def _lambda(mu, nu=0.49):
        E = 2* mu * (1 + nu)
        return (E*nu) / ((1 + nu) * (1 - 2*nu))



class TetwiseParameters(Parameters):
    
    @cached_property
    def NUM_OF_MATERIALS(self):
        return self.descriptor.scene.object.tetrahedra.shape[0]   # tet count

    @cached_property
    def initial_parameters(self):
        initial_tetwise_param_list = self.initial_param_list[:self.NUM_OF_PARAMS]*self.NUM_OF_MATERIALS
        return torch.tensor(initial_tetwise_param_list, dtype=torch.float32, device=self.descriptor.device).reshape(self.NUM_OF_MATERIALS, self.NUM_OF_PARAMS)
    
    @cached_property
    def tetwise_param_dist(self):
        return np.arange(self.NUM_OF_MATERIALS)

    


if __name__ == '__main__':
    torch.manual_seed(0)
    desc = Descriptor('./experiments/paper_experiments/lh_m+e/lh_m+e_elasticity_tetwise.exp')
    init_params = [6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0] 
    param = TetwiseParameters(desc, initial_parameters=init_params, optimizable=[0,1,0,0], no_perturb=True)
    print(param.distribute_material_parameters().shape)
    np.save('tetwise.npy', param.distribute_material_parameters().detach().cpu().numpy())
    # param.parameter_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    # print(param.material_tensor)
    # print(param.density_tensor)
    # print(param.parameter_tensor)
