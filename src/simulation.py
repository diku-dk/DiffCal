import dflex as df
from abc import ABC, abstractmethod, abstractproperty
import torch
import torch.utils.checkpoint

import numpy as np
from cached_property import cached_property

from src.descriptor import Descriptor
from src.parameters import Parameters, TetwiseParameters


class Simulation(ABC):

    def __init__(self, descriptor: Descriptor, parameters: Parameters, checkpoint: bool = True, checkpoint_max_steps: int = 20000):
        self.descriptor = descriptor
        self.parameters = parameters
        self.sim_steps = int(self.descriptor.experiment.duration / self.descriptor.experiment.dt)
        self.state = self.get_model().state()
        self.checkpoint = checkpoint
        self.checkpoint_max_steps = checkpoint_max_steps
    
    @abstractmethod
    def get_model(self) -> df.sim.Model:
        """ dFlex model. """
        pass

    @abstractproperty
    def _integrator(self) -> df.sim.SemiImplicitIntegrator:
        """ The integrator used to step through time. """
        pass

    @abstractmethod
    def run(self) -> df.sim.State:
        """ Runs the simulation
        :param checkpoint: Whether or not to use checkpointing.
        :param checkpoint_max_steps: Maximum number of steps before checkpointing should be applied.
        :return: The state of the simulation of 'sim_steps' steps."""
        pass



class StaticBendSimulation(Simulation):
        
    @cached_property
    def _initial_model(self) -> df.sim.Model:
        builder = df.sim.ModelBuilder()
        obj = self.descriptor.scene.object
        builder.add_soft_mesh(
            pos=obj.displacement * obj.scale,
            rot=obj.orientation,
            scale=obj.scale,
            vel=np.zeros(3, dtype=np.float32),
            vertices=obj.vertices,
            indices=obj.tetrahedra.reshape(-1),
            density=self.parameters.initial_density,  # 1080 kg / m^3
            k_mu=66e4,  # First Lamé parameter mu
            k_lambda=20e4,  # Second Lamé parameter lambda
            k_damp=5.0   # Damping parameter
        )
        builder.particle_mass = np.array(builder.particle_mass)
        builder.particle_mass[obj.dirichlet_boundary_particles] = 0.0
        builder.particle_mass = builder.particle_mass.tolist()
        model = builder.finalize(self.descriptor.device)
        model.tri_ke, model.tri_ka, model.tri_kd, model.tri_kb = 0.0, 0.0, 0.0, 0.0
        model.ground = False
        model.gravity = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float32).to(self.descriptor.device)
        # initial_particle_mass = model.particle_mass.detach().clone() / self.parameters.initial_density
        # initial_inv_particle_mass = model.particle_inv_mass.detach().clone() * self.parameters.initial_density
        # model.particle_mass = initial_particle_mass * self.parameters.density_tensor
        # model.particle_inv_mass = initial_inv_particle_mass / self.parameters.density_tensor
        return model

    def get_model(self):
        self._initial_model.tet_materials = self.parameters.distribute_material_parameters()
        return self._initial_model

    def _move_vertices(self):
        return None

    @cached_property
    def _integrator(self) -> df.sim.SemiImplicitIntegrator:
        return df.sim.SemiImplicitIntegrator()

    @property
    def use_checkpoint(self) -> bool:
        return self.checkpoint and (self.sim_steps > self.checkpoint_max_steps)

    def get_current_state(self):
        return self.get_model().state()
    
    def prepare_run(self):
        return None

    def run(self) -> df.sim.State:
        self.prepare_run()
        model = self.get_model()
        self.state = self.get_current_state()
        if self.use_checkpoint:
            def segmented_sim(sub_steps, state_q, dummy):
                self.state.q = state_q
                for _ in range(sub_steps):
                    self.state = self._integrator.forward(model, self.state, self.descriptor.experiment.dt)
                    self._move_vertices()
                return self.state.q

            num_iterations = int(self.sim_steps / self.checkpoint_max_steps)
            for _ in range(num_iterations):
                self.state.q = torch.utils.checkpoint.checkpoint(lambda state, dummy: segmented_sim(self.sim_steps // num_iterations, state, dummy),
                                                self.state.q.detach().clone().requires_grad_(True), torch.zeros(1, requires_grad=True))
            remaining_steps = self.sim_steps - num_iterations * (self.sim_steps // num_iterations)
            if remaining_steps != 0:
                self.state.q = torch.utils.checkpoint.checkpoint(lambda state, dummy: segmented_sim(remaining_steps, state, dummy),
                                                self.state.q.detach().clone().requires_grad_(True), torch.zeros(1, requires_grad=True))
        else:
            for _ in range(self.sim_steps):
                self.state = self._integrator.forward(model, self.state, self.descriptor.experiment.dt)
                self._move_vertices()
        return self.state


class DynamicBendSimulation(StaticBendSimulation):

    def get_current_state(self):
        return self.state


class StaticTwistSimulation(StaticBendSimulation):

    def __init__(self, descriptor: Descriptor, parameters: Parameters,
                checkpoint: bool = True, checkpoint_max_steps: int = 20000):

        super().__init__(descriptor, parameters, checkpoint, checkpoint_max_steps)
        num_steps = int(self.descriptor.experiment.duration // (1 / 30))

        self.interpolate_poses(vertex_positions=self._initial_model.state().q,
                                   num_steps=num_steps,
                                   bbox=torch.tensor([[-float('inf'), -0.0678],
                                                     [-float('inf'), float('inf')],
                                                      [-float('inf'), float('inf')]], dtype=torch.float, device=self.descriptor.device),
                                   rotation_a=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.descriptor.device),
                                   rotation_b=torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float, device=self.descriptor.device))
    def prepare_run(self):
            self.interpolation_duration = 0.5
            self.positional_update_step = 0
            moving_dirichlet_steps = self.sim_steps * self.interpolation_duration
            self.moving_dirichlet_substeps = int(moving_dirichlet_steps / self.positional_updates.shape[0])
            self.moving_dirichlet_step = 0

    def _move_vertices(self):
        if self.moving_dirichlet_step % self.moving_dirichlet_substeps == 0:            
            if self.positional_update_step < self.positional_updates.shape[0]:
                self.state.q[self.vertices_to_move] += self.positional_updates[self.positional_update_step]
                self.positional_update_step += 1
        self.moving_dirichlet_step += 1

    def interpolate_poses(self,
                          vertex_positions: torch.Tensor,
                          num_steps: int,
                          bbox: torch.Tensor,
                          position_a: torch.Tensor = None,
                          position_b: torch.Tensor = None,
                          rotation_a: torch.Tensor = None,
                          rotation_b: torch.Tensor = None):

        from src.utils.quat import Quaternion as Q
        import pytorch3d.transforms as pt
        if position_a is None: position_a = torch.zeros(3).float().to(self.descriptor.device)
        if position_b is None: position_b = torch.zeros(3).float().to(self.descriptor.device)
        if rotation_a is None: rotation_a = torch.tensor([1.0, 0.0, 0.0, 0.0]).float().to(self.descriptor.device)
        if rotation_b is None: rotation_b = torch.tensor([1.0, 0.0, 0.0, 0.0]).float().to(self.descriptor.device)
        # Create vertex mask
        mask = np.empty((0, 1), dtype=np.int)
        mean_pos = vertex_positions.mean(0)
        for vi, v in enumerate(vertex_positions):
            if bbox[0, 1] >= v[0] - mean_pos[0] >= bbox[0, 0] and\
               bbox[1, 1] >= v[1] - mean_pos[1] >= bbox[1, 0] and\
               bbox[2, 1] >= v[2] - mean_pos[2] >= bbox[2, 0]:
                mask = np.append(mask, vi)
        self.vertices_to_move = mask
        key_frames = torch.zeros((num_steps, mask.shape[0], 3), dtype=torch.float, device=self.descriptor.device)
        self.positional_updates = torch.zeros((num_steps, mask.shape[0], 3), dtype=torch.float, device=self.descriptor.device)
        vertex_positions = vertex_positions.clone()
        pivot_point = vertex_positions[mask].min(0)[0] + (vertex_positions[mask].max(0)[0] - vertex_positions[mask].min(0)[0]) / 2
        for i in range(num_steps):
            q = Q.torch_SLERP(rotation_a, rotation_b, i / float(num_steps))
            p = position_a + (position_a + position_b) * (i / float(num_steps))
            next_frame = torch.mm(vertex_positions[mask] - pivot_point,
                                  pt.quaternion_to_matrix(q)) + pivot_point
            next_frame = next_frame + p
            if i != 0:
                self.positional_updates[i] = next_frame - key_frames[i-1]
            key_frames[i, :, :] = next_frame


class DynamicTwistSimulation(StaticTwistSimulation):

    def get_current_state(self):
        return self.state






if __name__ == '__main__':
    desc = Descriptor('./experiments/paper_experiments/twist_e+m/twist_e+m_elasticity.exp')
    init_params = [6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0]
    param = Parameters(desc, initial_parameters=init_params, optimizable=[0,1,0,0], no_perturb=True)
    sim_static = StaticTwistSimulation(desc, param)
    state = sim_static.run().q
    print(state)