import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise


class TestStaticCamera(unittest.TestCase):

    target_path = 'tests/expected_output'

    def examine_target_images(self, exp: Experiment, target_tensor: str):
        x = exp.camera.target_images
        t = torch.load(f'{self.target_path}/{target_tensor}')
        self.assertTrue(torch.equal(x, t))

    def test_target_images(self):
        self.examine_target_images(lh_e, 'lh_e_target_images.pt')
        self.examine_target_images(spine_e, 'spine_e_target_images.pt')
        self.examine_target_images(lh_m, 'lh_m_target_images.pt')
        self.examine_target_images(spine_m, 'spine_m_target_images.pt')
        self.examine_target_images(lh_me, 'lh_me_target_images.pt')
        self.examine_target_images(twist_em, 'twist_em_target_images.pt')
        self.examine_target_images(lh_me_tetwise, 'lh_me_tetwise_target_images.pt')
        self.examine_target_images(twist_em_tetwise, 'twist_em_tetwise_target_images.pt')
        
if __name__ == '__main__':
    unittest.main()
