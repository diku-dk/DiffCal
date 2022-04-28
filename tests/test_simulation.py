
import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise


class TestStaticBendSimulation(unittest.TestCase):

    target_path = 'tests/expected_output'

    def examine_initial_state(self, exp: Experiment, target_tensor: str):
        x = exp.simulation.get_model().state().q
        t = torch.load(f'{self.target_path}/{target_tensor}')
        self.assertTrue(torch.allclose(x, t, atol=1e-5))

    def examine_final_state(self, exp: Experiment, target_tensor: str):
        x = exp.simulation.run().q
        t = torch.load(f'{self.target_path}/{target_tensor}')
        self.assertTrue(torch.allclose(x, t, atol=1e-5))

    def test_state(self):
        # self.examine_initial_state(lh_e, 'lh_e_initial_state.pt')        
        # self.examine_final_state(lh_e, 'lh_e_final_state.pt')

        # self.examine_initial_state(spine_e, 'spine_e_initial_state.pt')        
        # self.examine_final_state(spine_e, 'spine_e_final_state.pt')

        # self.examine_initial_state(lh_m, 'lh_m_initial_state.pt')        
        # self.examine_final_state(lh_m, 'lh_m_final_state.pt')

        # self.examine_initial_state(spine_m, 'spine_m_initial_state.pt')        
        # self.examine_final_state(spine_m, 'spine_m_final_state.pt')

        # self.examine_initial_state(lh_me, 'lh_me_initial_state.pt')        
        # self.examine_final_state(lh_me, 'lh_me_final_state.pt')

        # self.examine_initial_state(twist_em, 'twist_em_initial_state.pt')        
        # self.examine_final_state(twist_em, 'twist_em_final_state.pt')

        # self.examine_initial_state(lh_me_tetwise, 'lh_me_tetwise_initial_state.pt')        
        # self.examine_final_state(lh_me_tetwise, 'lh_me_tetwise_final_state.pt')

        self.examine_initial_state(twist_em_tetwise, 'twist_em_tetwise_initial_state.pt')        
        self.examine_final_state(twist_em_tetwise, 'twist_em_tetwise_final_state.pt')





        

# class TestDynamicBendSimulation(unittest.TestCase):

#     data_path = 'tests/expected_output'

#     @classmethod
#     def setUpClass(cls):
#         param_1 = Parameters(
#             Descriptor('./experiments/paper_experiments/lh_m/lh_m_viscosity_0.exp', set_seed=True),
#             initial_parameters=[9e4, 2.5e6, 5.0],
#             initial_density=1080.,
#             optimizable=[1,1,0,1],
#             no_perturb=True
#         )
#         cls.dynamic_sim_1 = DynamicBendSimulation(param_1.descriptor, param_1)

#         param_2 = Parameters(
#             Descriptor('./experiments/paper_experiments/lh_m/lh_m_viscosity_1.exp', set_seed=True),
#             initial_parameters=[7e4, 2.5e6, 5.0],
#             initial_density=1080.,
#             optimizable=[1,1,0,1],
#             no_perturb=True
#         )
#         cls.camera_1 = DynamicCamera(param_1.descriptor, background_threshold=0.549)
#         # cls.dynamic_sim_2 = DynamicBendSimulation(param_2.descriptor, param_2)

#     @classmethod
#     def tearDownClass(cls) -> None:
#         del cls.dynamic_sim_1
#         del cls.camera_1
        
#     def test_run(self):
#         final_state_1 = torch.load(f'{self.data_path}/final_state_dynamic_1.pt')
#         final_state_2 = torch.load(f'{self.data_path}/final_state_dynamic_2.pt')
#         for sub_steps in self.camera_1.data_based_sub_steps:
#             self.dynamic_sim_1.sim_steps = sub_steps
#             self.dynamic_sim_1.run()
#             # self.dynamic_sim_2.sim_steps = sub_steps
#             # self.dynamic_sim_2.run()
#         self.assertTrue(torch.allclose(self.dynamic_sim_1.state.q, final_state_1, atol=1e-5))
#         # self.assertTrue(torch.allclose(self.dynamic_sim_2.state.q, final_state_2, atol=1e-5))


# class TestStaticTwistSimulation(unittest.TestCase):

#     data_path = 'tests/expected_output'

#     @classmethod
#     def setUpClass(cls):
#         param_1 = Parameters(
#             Descriptor('./experiments/paper_experiments/twist_e+m/twist_e+m_elasticity.exp', set_seed=True),
#             initial_parameters=[6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0],
#             initial_density=1080.,
#             optimizable=[0,1,0,0],
#             no_perturb=True
#         )
#         cls.sim_static_1 = StaticTwistSimulation(param_1.descriptor, param_1)

#     @classmethod
#     def tearDownClass(cls) -> None:
#         del cls.sim_static_1        

#     def test_get_final_state(self):
#         final_state_1 = torch.load(f'{self.data_path}/static_twist_simulation_1.pt')
#         self.assertTrue(torch.allclose(self.sim_static_1.run().q, final_state_1, atol=1e-5))
        
if __name__ == '__main__':
    unittest.main()