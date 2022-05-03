import tqdm
from abc import ABC, abstractmethod
import torch
from typing import Union, Tuple
<<<<<<< HEAD
import os
=======
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6

from src.descriptor import Descriptor
from src.parameters import Parameters, TetwiseParameters
from src.loss import Loss, L2Loss, TetwiseL2Loss, TwoModeL2Loss, TwoModeTetwiseL2Loss
from src.simulation import StaticBendSimulation, StaticTwistSimulation
from src.render import StaticRender, DynamicRender
from src.camera import StaticCamera, DynamicCamera

class Minimizer(ABC):

<<<<<<< HEAD
    def __init__(self, loss: Loss, num_iters: int = 10, lr: Union[float, torch.Tensor]=1e3, eps: float = 1e-8, save_results=True):
=======
    def __init__(self, loss: Loss, num_iters: int = 10, lr: Union[float, torch.Tensor]=1e3, eps: float = 1e-8):
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
        self.loss = loss
        self.num_iters = num_iters
        self.lr = lr
        self.eps = eps
<<<<<<< HEAD
        self.save_results=save_results
        if save_results:
            path = 'results'
            self.rep_number = len(os.listdir(path))
            self.saving_path = f'{path}/rep_{self.rep_number}'
            
=======
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
    
    @abstractmethod
    def minimize(self):
        pass
<<<<<<< HEAD
=======

    # def save_log(self, path):
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
        


class AdamManual(Minimizer):
    
    def __init__(self,
        loss: Loss,
        num_iters: int = 10,
        lr: Union[float, torch.Tensor] = 1e3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
<<<<<<< HEAD
        delta: float = 1e-16,
        save_results=True):

        super().__init__(loss, num_iters, lr, eps, save_results)
        self.betas = betas
        self.delta = delta

=======
        delta: float = 1e-16):

        super().__init__(loss, num_iters, lr, eps)
        self.betas = betas
        self.delta = delta
        self.log = {}
        self.optimal_loss = None
        self.optimal_parameters = None
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
    
    def minimize(self):
        current_param = self.loss.parameters.parameter_tensor
        m = torch.zeros_like(current_param)
        v = torch.zeros_like(current_param)
        iteration_range = tqdm.tqdm(range(1, self.num_iters))
        
        for i in iteration_range:
<<<<<<< HEAD
            if self.save_results:
                os.makedirs(f'{self.saving_path}/parameters', exist_ok=True)
                torch.save(current_param, f'{self.saving_path}/parameters/p_{i}.pt')
=======
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
            loss, grad = self.loss.value, self.loss.grad
            beta_i = self.betas[0] * (1. - i / self.num_iters) / ((1 + self.betas[0]) + self.betas[0] * (1 - i / self.num_iters))
            m = grad + beta_i * m
            v = self.betas[1] * v + (1 - self.betas[1]) * grad**2
            m_hat = m / (1 - self.betas[0]**i)
            v_hat = v / (1 - self.betas[1]**i)
               
            next_param = current_param - self.lr * m_hat / torch.sqrt(v_hat + 1e-8)
            # max_tensor = torch.tensor([100.] + [5e3, 1e4, 5.0] * ((len(current_param) - 1) // 3), dtype=torch.float32, device='cuda')
            # min_tensor = -max_tensor
            # next_param = current_param - torch.max(torch.min(self.lr * m_hat / (torch.sqrt(v_hat + 1e-8)), max_tensor), min_tensor)

            self.loss.parameters.parameter_tensor = next_param
            if (torch.norm(current_param - next_param) < self.delta) or (torch.norm(grad) < self.eps):
<<<<<<< HEAD
                break
            else:
                current_param = next_param
            
=======
                self.optimal_parameters = next_param
                self.optimal_loss = loss
                break
            else:
                current_param = next_param
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
        return current_param, loss


class AdamTorch(Minimizer):

    def minimize(self):
        adam = torch.optim.Adam([self.loss.parameters.material_tensor], lr=self.lr)
        for _ in range(self.num_iters):
            adam.zero_grad()
            loss = self.loss.value
            loss.backward()
            adam.step()
            print(loss)
            print(self.loss.parameters.material_tensor)





if __name__ == '__main__':
    path = './experiments/paper_experiments/lh_m+e_2_mode'
    desc_1 = Descriptor(f'{path}/lh_m+e_elasticity_tetwise.exp', device='cuda', set_seed=True)
    desc_2 = Descriptor(f'{path}/twist_e+m_elasticity_tetwise.exp', device='cuda', set_seed=True)
    camera_1, camera_2 = StaticCamera(desc_1), StaticCamera(desc_2)
    init_params = [6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0]
    
    params = TetwiseParameters(desc_1, initial_parameters=init_params, optimizable=[0,1,0,0], no_perturb=True)
    sim_1, sim_2 = StaticBendSimulation(desc_1, params), StaticTwistSimulation(desc_2, params)
    render_1, render_2 = StaticRender(desc_1, sim_1, camera_1), StaticRender(desc_2, sim_2, camera_2)
    loss = TwoModeTetwiseL2Loss(params, desc_1, render_1, camera_1, desc_2, render_2, camera_2)

    AdamManual(loss, lr=1000).minimize()