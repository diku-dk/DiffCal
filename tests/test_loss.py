
import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import h_e_c, h_e_s, h_m_c, h_m_s, h_em_c, t_em_c, h_em_c_tetwise, t_em_c_tetwise


class TestLoss(unittest.TestCase):

    target_path = 'tests/expected_output'

    def is_initial_loss_close(self, exp: Experiment, target_tensor: str):
        actual_output = exp.loss.value
        expected_output = torch.load(f'{self.target_path}/{target_tensor}')
        return self.assertTrue(torch.allclose(expected_output, actual_output, atol=1e3))

    def is_initial_grad_close(self, exp: Experiment, target_tensor: str):
        actual_output = exp.loss.grad
        expecte_output = torch.load(f'{self.target_path}/{target_tensor}')
        return self.assertTrue(torch.allclose(expecte_output, actual_output, atol=1e-4))

    def test_loss_h_e_c(self):
        self.is_initial_loss_close(h_e_c, 'h_e_c_initial_loss.pt')

    def test_loss_h_e_s(self):
        self.is_initial_loss_close(h_e_s, 'h_e_s_initial_loss.pt')

    def test_loss_h_m_c(self):
        self.is_initial_loss_close(h_m_c, 'h_m_c_initial_loss.pt')

    def test_loss_h_m_s(self):
        self.is_initial_loss_close(h_m_s, 'h_m_s_initial_loss.pt')

    def test_loss_h_em_c(self):
        self.is_initial_loss_close(h_em_c, 'h_em_c_initial_loss.pt')

    def test_loss_t_em_c(self):
        self.is_initial_loss_close(t_em_c, 't_em_c_initial_loss.pt')

    def test_loss_h_em_c_tetwise(self):
        self.is_initial_loss_close(h_em_c_tetwise, 'h_em_c_tetwise_initial_loss.pt')

    def test_loss_h_em_c_tetwise(self):
        self.is_initial_loss_close(t_em_c_tetwise, 't_em_c_tetwise_initial_loss.pt')

    def test_grad_h_e_c(self):
        self.is_initial_grad_close(h_e_c, 'h_e_c_initial_grad.pt')

    def test_grad_h_e_s(self):
        self.is_initial_grad_close(h_e_s, 'h_e_s_initial_grad.pt')

    def test_grad_h_m_c(self):
        self.is_initial_grad_close(h_m_c, 'h_m_c_initial_grad.pt')
    
    def test_grad_h_m_s(self):
        self.is_initial_grad_close(h_m_s, 'h_m_s_initial_grad.pt')
    
    def test_grad_h_em_c(self):
        self.is_initial_grad_close(h_em_c, 'h_em_c_initial_grad.pt')

    def test_grad_t_em_c(self):
        self.is_initial_grad_close(t_em_c, 't_em_c_initial_grad.pt')
    
    def test_grad_h_em_c_tetwise(self):
        self.is_initial_grad_close(h_em_c_tetwise, 'h_em_c_tetwise_initial_grad.pt')

    def test_grad_h_em_c_tetwise(self):
        self.is_initial_grad_close(t_em_c_tetwise, 't_em_c_tetwise_initial_grad.pt')



if __name__ == '__main__':
    unittest.main()
