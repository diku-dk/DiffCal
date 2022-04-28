import unittest
import torch

from src.experiment_types import Experiment
from tests.cases import lh_e, spine_e, lh_m, spine_m, lh_me, twist_em, lh_me_tetwise, twist_em_tetwise, lh_e_dynamic


class TestStaticRender(unittest.TestCase):

    target_path = 'tests/expected_output'

    def examine_rendered_images(self, exp: Experiment, target_tensor: str):
        x = exp.render.rendered_images
        t = torch.load(f'{self.target_path}/{target_tensor}')
        print(x.shape)
        print(t.shape)
        print('***********', torch.sum((x-t)**2))
        self.assertTrue(torch.sum((x-t)**2)<1e-2)

    def test_rendered_images(self):
        # self.examine_rendered_images(lh_e, 'lh_e_rendered_images.pt')
        # self.examine_rendered_images(spine_e, 'spine_e_rendered_images.pt')
        # self.examine_rendered_images(lh_m, 'lh_m_rendered_images.pt')
        # self.examine_rendered_images(spine_m, 'spine_m_rendered_images.pt')
        # self.examine_rendered_images(lh_me, 'lh_me_rendered_images.pt')
        # self.examine_rendered_images(twist_em, 'twist_em_rendered_images.pt')
        # self.examine_rendered_images(lh_me_tetwise, 'lh_me_tetwise_rendered_images.pt')
        self.examine_rendered_images(twist_em_tetwise, 'twist_em_tetwise_rendered_images.pt')

        # self.examine_rendered_images(lh_e_dynamic, 'lh_e_dynamic_rendered_images.pt')



# class TestDynamicRender:
# # (unittest.TestCase):

#     data_path = 'tests/expected_output'

#     def setUp(self):
#         param_1 = Parameters(Descriptor('./experiments/paper_experiments/lh_m/lh_m_viscosity_1.exp', set_seed=True, device='cpu'),
#             initial_parameters=[7e4, 2.5e6, 5.0],
#             initial_density=1080.,
#             optimizable=[1,1,0,1],
#             no_perturb=True
#         )
#         sim_static_1 = DynamicBendSimulation(param_1.descriptor, param_1)
#         camera_1 = DynamicCamera(param_1.descriptor, background_threshold=0.549)
#         self.render_1 = DynamicRender(param_1.descriptor, sim_static_1, camera_1)
    
#     def tearDown(self) -> None:
#         del self.render_1

#     def test_rendered_images(self):
#         import matplotlib.pyplot as plt
#         rendered_images_1 = torch.load(f'{self.data_path}/dynamic_rendered_images_2.pt')
#         i = 0
#         print(torch.allclose(self.render_1.rendered_images, rendered_images_1, atol=1e-3))
#         for img_tr, img_pr in zip(rendered_images_1, self.render_1.rendered_images):
#             i += 1
#             plt.figure(figsize=(30,7))
#             plt.suptitle(f'Device: {self.render_1.descriptor.device}')
#             plt.subplot(131)
#             plt.imshow((img_tr.detach().cpu().numpy()*255).astype('uint8').squeeze())
#             plt.title('Max')
#             plt.subplot(132)
#             plt.imshow((img_pr.detach().cpu().numpy()*255).astype('uint8').squeeze())
#             plt.title('Kasra')
#             plt.subplot(133)
#             img = plt.imshow((((img_tr-img_pr)**2).detach().cpu().numpy()*255).astype('uint8').squeeze())
#             plt.title('Difference^2')
#             plt.show()
#             # plt.savefig(f'img_{i}.png')

if __name__ == '__main__':
    unittest.main()
