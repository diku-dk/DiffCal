import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import h_e_c, h_e_s, h_m_c, h_m_s, h_em_c, t_em_c, h_em_c_tetwise, t_em_c_tetwise

class TestParameters(unittest.TestCase):

    target_path = 'tests/expected_output'

    def are_distribute_material_parameters_equal(self, exp: Experiment, target_tensor: str):
        actual_output = exp.parameters.distribute_material_parameters()
        expected_output = torch.load(f'{self.target_path}/{target_tensor}')
        return self.assertTrue(torch.equal(expected_output, actual_output))

    def test_distribute_material_parameters_h_e_c(self):
        self.are_distribute_material_parameters_equal(h_e_c, 'h_e_c_distribute_material_parameters.pt')

    def test_distribute_material_parameters_h_e_s(self):
        self.are_distribute_material_parameters_equal(h_e_s, 'h_e_s_distribute_material_parameters.pt')

    def test_distribute_material_parameters_h_m_c(self):
        self.are_distribute_material_parameters_equal(h_m_c, 'h_m_c_distribute_material_parameters.pt')

    def test_distribute_material_parameters_h_m_s(self):
        self.are_distribute_material_parameters_equal(h_m_s, 'h_m_s_distribute_material_parameters.pt')

    def test_distribute_material_parameters_h_em_c(self):
        self.are_distribute_material_parameters_equal(h_em_c, 'h_em_c_distribute_material_parameters.pt')

    def test_distribute_material_parameters_t_em_c(self):
        self.are_distribute_material_parameters_equal(t_em_c, 't_em_c_distribute_material_parameters.pt')

    def test_distribute_material_parameters_t_em_c_tetwise(self):
        self.are_distribute_material_parameters_equal(h_em_c_tetwise, 'h_em_c_tetwise_distribute_material_parameters.pt')

    def test_distribute_material_parameters_t_em_c_tetwise(self):
        self.are_distribute_material_parameters_equal(t_em_c_tetwise, 't_em_c_tetwise_distribute_material_parameters.pt')

        


if __name__ == '__main__':
    unittest.main()