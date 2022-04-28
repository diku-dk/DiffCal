import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise

class TestParameters(unittest.TestCase):

    target_path = 'tests/expected_output'

    def examine_distribute_material_parameters(self, exp: Experiment, target_tensor: str):
        torch.set_printoptions(precision=10)
        x = exp.parameters.distribute_material_parameters()
        self.assertIsInstance(x, torch.Tensor)
        self.assertTrue(x.dtype, torch.float32)
        t = torch.load(f'{self.target_path}/{target_tensor}')
        print(x)
        print(t)
        self.assertTrue(torch.equal(x, t))

    def test_distribute_material_parameters(self):
        # self.examine_distribute_material_parameters(lh_e, 'lh_e_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(spine_e, 'spine_e_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(lh_m, 'lh_m_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(spine_m, 'spine_m_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(lh_me, 'lh_me_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(twist_em, 'twist_em_distribute_material_parameters.pt')
        # self.examine_distribute_material_parameters(lh_me_tetwise, 'lh_me_tetwise_distribute_material_parameters.pt')
        self.examine_distribute_material_parameters(twist_em_tetwise, 'twist_em_tetwise_distribute_material_parameters.pt')

        


if __name__ == '__main__':
    unittest.main()