import ast
import torch
import pytorch3d.transforms as pt
import numpy as np
from typing import Optional, List


from src.utils.object import Object

"""
Scene Descriptor specification (.sdesc):
========================================
duration[float] sub_steps[int] fps[int] num_objects[int] actuated_objects[List[Int]]
(0) Objfile[str] Tetfile[str] scale[float] X-displacement[float] Y-displacement[float] Z-displacement[float] quat[float x 4] actuationfile[Optional[str]] normalize[bool]
(1) Objfile[str] Tetfile[str] scale[float] X-displacement[float] Y-displacement[float] Z-displacement[float] quat[float x 4] actuationfile[Optional[str]] normalize[bool]
...
(N) Objfile[str] Tetfile[str] scale[float] X-displacement[float] Y-displacement[float] Z-displacement[float] quat[float x 4] actuationfile[Optional[str]] normalize[bool]
(0) (((dirichlet_boundary_x0,
       dirichlet_boundary_x1),
      (dirichlet_boundary_y0,
       dirichlet_boundary_y1),
      (dirichlet_boundary_z0,
       dirichlet_boundary_z1)), // Boundary 0
     ((dirichlet_boundary_x2,
       dirichlet_boundary_x3),
      (dirichlet_boundary_y2,
       dirichlet_boundary_y3),
      (dirichlet_boundary_z2,
       dirichlet_boundary_z3)), // Boundary 1
       ...
       )
...
(N) (((dirichlet_boundary_x0,
       dirichlet_boundary_x1),
      (dirichlet_boundary_y0,
       dirichlet_boundary_y1),
      (dirichlet_boundary_z0,
       dirichlet_boundary_z1)), // Boundary 0
     ((dirichlet_boundary_x2,
       dirichlet_boundary_x3),
      (dirichlet_boundary_y2,
       dirichlet_boundary_y3),
      (dirichlet_boundary_z2,
       dirichlet_boundary_z3)), // Boundary 1
       ...
       )
"""


class SceneDescriptor:
    duration: float
    """Simulation duration in seconds."""
    sub_steps: int
    """Sub-steps to perform between each frame."""
    fps: int
    """Frames pr. second of the simulation."""
    dt: float
    """Time between each simulation step."""
    sim_steps: int
    """Total number of simulation step to perform."""
    scale: float
    """Scale of objects in the scene."""
    surface_files: List[str]
    """List of surface mesh files."""
    volume_files: List[str]
    """List of volume mesh files."""
    objects: List[Object] = []
    """List of Scene Objects."""

    def __init__(self,
                 duration: float = 1.0,
                 sub_steps: int = 32,
                 fps: int = 60,
                 surface_file: Optional[str] = None,
                 volume_file: Optional[str] = None,
                 object_transform: Optional[np.ndarray] = None,
                 scale: Optional[float] = 0.01):
        """
        Initializes a SceneDescriptor object.
        :param duration: Duration of the simulation.
        :param sub_steps: Number of steps to perform between each frame.
        :param fps: Number of frames pr. second.
        :param surface_files: List of surface mesh files.
        :param volume_files: List of volume mesh files.
        """
        self.duration = duration
        self.sub_steps = sub_steps
        self.fps = fps
        self.dt = (1.0 / fps) / self.sub_steps
        self.sim_steps = int(self.duration / self.dt)
        self.scale = scale

        self.surface_file = surface_file if not (surface_file is None) else []
        self.volume_file = volume_file if not (volume_file is None) else []
        if object_transform is not None:
            self.objects_transform = object_transform
        else:
            self.objects_transform = [np.eye(4, dtype=np.float32) for _ in self.volume_file]
        self.object = Object(surface_file=self.surface_file,
                      volume_file=self.volume_file,
                      scale=self.scale,
                      displacement=self.objects_transform[:3, 3],
                      orientation=pt.matrix_to_quaternion(torch.from_numpy(self.objects_transform[:3, :3])).float().numpy(),
                      normalize=True)