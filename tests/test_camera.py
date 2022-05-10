import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import h_e_c, h_e_s, h_m_c, h_m_s, h_em_c, t_em_c, h_em_c_tetwise, t_em_c_tetwise


class TestStaticCamera(unittest.TestCase):

    target_path = 'tests/expected_output'

    def are_target_images_equal(self, exp: Experiment, target_tensor: str):
        actual_output = exp.camera.target_images
        expected_output = torch.load(f'{self.target_path}/{target_tensor}')
        return self.assertTrue(torch.equal(expected_output, actual_output))

    def test_target_images_h_e_c(self):
        self.are_target_images_equal(h_e_c, 'h_e_c_target_images.pt')
    
    def test_target_images_h_e_s(self):
        self.are_target_images_equal(h_e_s, 'h_e_s_target_images.pt')
    
    def test_target_images_h_m_c(self):
        self.are_target_images_equal(h_m_c, 'h_m_c_target_images.pt')

    def test_target_images_h_m_s(self):
        self.are_target_images_equal(h_m_s, 'h_m_s_target_images.pt')

    def test_target_images_h_em_c(self):    
        self.are_target_images_equal(h_em_c, 'h_em_c_target_images.pt')
    
    def test_target_images_t_em_c(self):
        self.are_target_images_equal(t_em_c, 't_em_c_target_images.pt')

    def test_target_images_h_em_c_tetwise(self):
        self.are_target_images_equal(h_em_c_tetwise, 'h_em_c_tetwise_target_images.pt')

    def test_target_images_t_em_c_tetwise(self):
        self.are_target_images_equal(t_em_c_tetwise, 't_em_c_tetwise_target_images.pt')
        
if __name__ == '__main__':
    unittest.main()
