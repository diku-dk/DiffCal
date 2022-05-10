import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import h_e_c, h_e_s, h_m_c, h_m_s, h_em_c, t_em_c, h_em_c_tetwise, t_em_c_tetwise


class TestStaticRender(unittest.TestCase):

    target_path = 'tests/expected_output'

    def are_rendered_images_close(self, exp: Experiment, target_tensor: str):
        x = exp.render.rendered_images
        t = torch.load(f'{self.target_path}/{target_tensor}')
        return self.assertTrue(torch.sum((x-t)**2)<1e-2)

    def test_rendered_images_h_e_c(self):
        self.are_rendered_images_close(h_e_c, 'h_e_c_rendered_images.pt')

    def test_rendered_images_h_e_s(self):
        self.are_rendered_images_close(h_e_s, 'h_e_s_rendered_images.pt')

    def test_rendered_images_h_m_c(self):
        self.are_rendered_images_close(h_m_c, 'h_m_c_rendered_images.pt')

    def test_rendered_images_h_m_s(self):
        self.are_rendered_images_close(h_m_s, 'h_m_s_rendered_images.pt')

    def test_rendered_images_h_em_c(self):
        self.are_rendered_images_close(h_em_c, 'h_em_c_rendered_images.pt')

    def test_rendered_images_t_em_c(self):
        self.are_rendered_images_close(t_em_c, 't_em_c_rendered_images.pt')

    def test_rendered_images_h_em_c_tetwise(self):
        self.are_rendered_images_close(h_em_c_tetwise, 'h_em_c_tetwise_rendered_images.pt')

    def test_rendered_images_t_em_c_tetwise(self):
        self.are_rendered_images_close(t_em_c_tetwise, 't_em_c_tetwise_rendered_images.pt')

if __name__ == '__main__':
    unittest.main()
