import numpy as np
import os
import torch
from typing import List
from cached_property import cached_property
from abc import ABC, abstractproperty

from src.descriptor import Descriptor


class Camera(ABC):

    def __init__(self, descriptor: Descriptor, background_threshold: float = 1.):
        self.descriptor = descriptor
        self.background_threshold = background_threshold


    @abstractproperty
    def target_images(self):
        pass

    @cached_property
    def _all_target_images(self) -> List[torch.Tensor]:
        """ The ground truth images we wish to replicate.
        """
        targets = []
        for timestamp in self._timestamps:
            target = np.load(f"{self.descriptor.experiment.camera_data_path}/d_{timestamp}.npy")
            target = torch.from_numpy(target).float().to(self.descriptor.device)
            if len(target.shape) != 3:
                target = target.reshape((target.shape[0], target.shape[1], 1))
            targets.append(target)
        for i, target in enumerate(targets):
            targets[i][target > self.background_threshold] = 0.0
        return targets
    
    @property
    def view_transforms(self) -> List[torch.Tensor]:
        """ View transforms extracted from the camera data.
        """
        view_transforms = []
        for view_transform in self._original_views:
            view_transform = np.dot(self.descriptor.experiment.rgb_to_depth_transform.T, view_transform)
            view_transform = np.dot(self.descriptor.experiment.camera_transform, view_transform)
            view_transform = torch.inverse(torch.from_numpy(view_transform).float().to(self.descriptor.device))
            view_transforms.append(view_transform)
        return view_transforms
        
    @property
    def data_based_sub_steps(self):
        timestamps = np.array(self._timestamps) - self._timestamps[0]
        # return [int((t - timestamps[ti]) / self.descriptor.experiment.dt) for ti, t in enumerate(timestamps[1:])]
        return [(int(((1. / 30.) / self.experiment_descriptor.dt))) for ti, t in enumerate(timestamps[1:]) ]


    @property
    def _target_filenames(self):  
        return [f for f in os.listdir(self.descriptor.experiment.camera_data_path) if 'd' in f]

    @property
    def _timestamps(self):
        return sorted([float(f.split('_')[1].split('.npy')[0]) for f in self._target_filenames])# [:12] # TODO: Find out the reason for 12.

    @property
    def _original_views(self) -> List[np.ndarray]:
        """ The untransformed view transforms.
        """
        original_views = []
        for timestamp in self._timestamps:
            original_view = np.load(f"{self.descriptor.experiment.camera_data_path}/v_0_{timestamp}.npy")
            original_views.append(original_view)
        return original_views


class StaticCamera(Camera):

    def __init__(self, descriptor: Descriptor, background_threshold: float = 1., view: int = -1):
        super().__init__(descriptor, background_threshold)
        self.view = view

    @cached_property
    def target_images(self):
        return torch.stack([self._all_target_images[self.view]]).to(self.descriptor.device, dtype=torch.float32)



class DynamicCamera(Camera):

    @cached_property
    def target_images(self):
        return torch.stack(self._all_target_images).to(self.descriptor.device)


    


