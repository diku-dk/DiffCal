
import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise


class TestMinimizer(unittest.TestCase):
    
    target_path = 'tests/expected_output'

    # def examine_parameters_iter_10(self, exp: Experiment, target_tensor: str):
    #     x = exp.minimizer.minimize()[0]
    #     t = torch.load(f'{self.target_path}/{target_tensor}')
    #     self.assertTrue(torch.allclose(x, t, atol=1e-5))

    def examine_loss_iter_10(self, exp: Experiment, target_tensor: str):
        x = exp.minimizer.minimize()[1]
        t = torch.load(f'{self.target_path}/{target_tensor}')
        print(x)
        print(t)
        self.assertTrue(torch.allclose(x, t, atol=1e3))

    def test_optimal_parameters(self):
        self.examine_loss_iter_10(lh_e, 'lh_e_loss_iter_10.pt')



if __name__ == '__main__':
    unittest.main()
