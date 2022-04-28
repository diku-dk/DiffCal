
from cached_property import cached_property
import os

from src.utils.experiment_descriptor import ExperimentDescriptor
from src.utils.scene_descriptor import SceneDescriptor

class Descriptor:

    def __init__(self, exp_file: str, device: str = 'cuda', seed: bool = False):
        self.exp_file = exp_file
        self.device = device
        self.seed = seed
        self._check_input()


    @cached_property
    def experiment(self) -> ExperimentDescriptor:
        """Contains all the information about the experiment about to be executed.
        """
        return ExperimentDescriptor().load(self.exp_file)

    @cached_property
    def scene(self) -> SceneDescriptor:
        """Contains all the information about objects in the scene.
        """
        for file in os.listdir(self.experiment.object_data_path):
            if file.endswith('.obj'): object_surface_file = f"{self.experiment.object_data_path}/{file}"
            if file.endswith('.tet'): object_volume_file = f"{self.experiment.object_data_path}/{file}"
        return SceneDescriptor(self.experiment.duration,
                                self.experiment.sub_steps,
                                self.experiment.fps,
                                object_surface_file,
                                object_volume_file,
                                self.experiment.object_transform)

    
    def _check_input(self):
        if not self.exp_file.endswith('.exp'):
            raise ValueError("exp_file must be of .exp format.")

    def set_seed(self, seed):
        import torch
        import numpy as np
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True