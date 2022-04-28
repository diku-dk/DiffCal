from typing import Union, List
import numpy as np


class ExperimentDescriptor:

    def __init__(self,
                 experiment_out_path: str = None,
                 duration: float = None,
                 dt: float = None,
                 sub_steps: int = None,
                 fps: int = None,
                 camera_data_path: str = None,
                 camera_intrinsics: Union[np.ndarray, str] = np.zeros((3, 3), dtype=np.float32),
                 camera_distortion: Union[np.ndarray, str] = np.zeros(14, dtype=np.float32),
                 rgb_to_depth_transform: Union[np.ndarray, str] = np.eye(4, dtype=np.float32),
                 camera_transform: np.ndarray = np.eye(4, dtype=np.float32)):
        self.camera_data_path = camera_data_path
        self.experiment_out_path = experiment_out_path
        self.duration = duration
        self.dt = dt
        self.sub_steps = sub_steps
        self.fps = fps
        self.camera_intrinsics = camera_intrinsics if type(camera_intrinsics) == np.ndarray else np.load(camera_intrinsics)
        self.camera_distortion = camera_distortion if type(camera_distortion) == np.ndarray else np.load(camera_distortion)
        self.rgb_to_depth_transform = rgb_to_depth_transform if type(rgb_to_depth_transform) == np.ndarray else np.load(rgb_to_depth_transform)
        self.camera_transform = camera_transform

    def save(self, out_path: str):
        with open(out_path, 'w') as fh:
            fh.write('EXPERIMENT SETTINGS\n')
            fh.write('experiment_out_path={0}\n'.format(self.experiment_out_path))
            fh.write('duration={0}\n'.format(self.duration))
            fh.write('dt={0}\n'.format(self.dt))
            fh.write('sub_steps={0}\n'.format(self.sub_steps))
            fh.write('fps={0}\n'.format(self.fps))
            fh.write('camera_data_path={0}\n'.format(self.camera_data_path))
            fh.write('camera_intrinsics={0}\n'.format(self.camera_intrinsics.flatten().tolist()))
            fh.write('camera_distortion={0}\n'.format(self.camera_distortion.flatten().tolist()))
            fh.write('rgb_to_depth_transform={0}\n'.format(self.rgb_to_depth_transform.flatten().tolist()))
            fh.write('camera_transform={0}\n'.format(self.camera_transform.flatten().tolist()))
            fh.write('object_data_path={0}\n'.format(self.object_data_path))
            fh.write('object_transform={0}\n'.format(self.object_transform.flatten().tolist()))
            fh.write('object_dirichlet_boundary_conditions={0}\n'.format(self.object_dirichlet_boundary_conditions.flatten().tolist()))
            fh.write('object_parameter_bounding_boxes={0}\n'.format(self.object_parameter_bounding_boxes.flatten().tolist()))

    @property
    def np_array_shapes(self):
        shapes = {}
        shapes['camera_intrinsics'] = (3,3)
        shapes['camera_distortion'] = (-1)
        shapes['rgb_to_depth_transform'] = (4,4)
        shapes['camera_transform'] = (4,4)
        shapes['object_transform'] = (4,4)
        shapes['object_dirichlet_boundary_conditions'] = (-1,3,2)
        shapes['object_parameter_bounding_boxes'] = (-1,3,2)
        return shapes

    @property
    def input_types(self):
        types = {key: np.ndarray for key in self.np_array_shapes.keys()}
        types['experiment_out_path'], types['camera_data_path'], types['object_data_path'] = str, str, str
        types['duration'], types['dt'] = float, float
        types['sub_steps'], types['fps'] = int, int
        return types
    
    def convert_type(self, key, value):
        input_types = self.input_types
        if input_types[key] == int:
            out = int(value)
        elif input_types[key] == float:
            out = float(value)
        elif input_types[key] == np.ndarray:
            as_list = [float(i) for i in value[1:-1].split(',')]
            out = np.array(as_list).reshape(self.np_array_shapes[key])
        else: # type str
            out = value
        return out 

    
    def load(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        settings = {line.split('=')[0]: line.split('=')[1][:-1] for line in lines[1:]}
        for key, value in settings.items():
            converted_value = self.convert_type(key, value)
            setattr(self, key, converted_value)
        return self

if __name__ == '__main__':
    exp_file = './lh_m+e_new.exp'
    exp = ExperimentDescriptor().load(exp_file)
    print(exp.__dict__)
    print(len(exp.__dict__))
