
import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise


class TestLoss(unittest.TestCase):

    target_path = 'tests/expected_output'

    def examine_initial_loss(self, exp: Experiment, target_tensor: str):
        x = exp.loss.value
        t = torch.load(f'{self.target_path}/{target_tensor}')
        print(x)
        print(t)
        self.assertTrue(torch.allclose(x, t, atol=1e3))

    def examine_initial_grad(self, exp: Experiment, target_tensor: str):
        x = exp.loss.grad
        t = torch.load(f'{self.target_path}/{target_tensor}')
        self.assertTrue(torch.allclose(x, t, atol=1e-4))

    def test_initial_loss_and_grad(self):
        # self.examine_initial_loss(lh_e, 'lh_e_initial_loss.pt')
        # self.examine_initial_grad(lh_e, 'lh_e_initial_grad.pt')

        # self.examine_initial_loss(spine_e, 'spine_e_initial_loss.pt')
        # self.examine_initial_grad(spine_e, 'spine_e_initial_grad.pt')

        # self.examine_initial_loss(lh_m, 'lh_m_initial_loss.pt')
        # self.examine_initial_grad(lh_m, 'lh_m_initial_grad.pt')

        # self.examine_initial_loss(spine_m, 'spine_m_initial_loss.pt')
        # self.examine_initial_grad(spine_m, 'spine_m_initial_grad.pt')

        # self.examine_initial_loss(lh_me, 'lh_me_initial_loss.pt')
        # self.examine_initial_grad(lh_me, 'lh_me_initial_grad.pt')

        # self.examine_initial_loss(twist_em, 'twist_em_initial_loss.pt')
        # self.examine_initial_grad(twist_em, 'twist_em_initial_grad.pt')

        # self.examine_initial_loss(lh_me_tetwise, 'lh_me_tetwise_initial_loss.pt')
        # self.examine_initial_grad(lh_me_tetwise, 'lh_me_tetwise_initial_grad.pt')

        self.examine_initial_loss(twist_em_tetwise, 'twist_em_tetwise_initial_loss.pt')
        self.examine_initial_grad(twist_em_tetwise, 'twist_em_tetwise_initial_grad.pt')
    


# class TestTetwiseLoss(unittest.TestCase):

#     data_path = 'tests/expected_output'

#     @classmethod
#     def setUpClass(cls):
#         param_1 = TetwiseParameters(Descriptor('./experiments/paper_experiments/lh_m+e/lh_m+e.exp', set_seed=True),
#             initial_parameters=[6e4, 2.5e6, 5.0, 6e4, 2.5e6, 5.0],
#             initial_density=1080.,
#             optimizable=[0,1,0,0],
#             no_perturb=True
#         )
#         sim_static_1 = StaticBendSimulation(param_1.descriptor, param_1)
#         camera_1 = StaticCamera(param_1.descriptor, background_threshold=0.520)
#         render_1 = StaticRender(param_1.descriptor, sim_static_1, camera_1)
#         cls.loss_1 = TetwiseL2Loss(param_1.descriptor, param_1, render_1, camera_1)
    
#     @classmethod
#     def tearDownClass(cls) -> None:
#         del cls.loss_1

#     def test_reg_loss(self):
#         value_1 = torch.load(f'{self.data_path}/reg_loss_1.pt')
#         self.assertTrue(torch.equal(self.loss_1.regularization_loss, value_1))


if __name__ == '__main__':
    unittest.main()
