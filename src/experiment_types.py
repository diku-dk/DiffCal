from abc import ABC, abstractmethod, abstractproperty
from cached_property import cached_property
from typing import List

from src.descriptor import Descriptor
from src.parameters import Parameters, TetwiseParameters
from src.simulation import Simulation, StaticBendSimulation, StaticTwistSimulation, DynamicBendSimulation, DynamicTwistSimulation
from src.render import Render, StaticRender, DynamicRender
from src.camera import Camera, StaticCamera, DynamicCamera
from src.loss import Loss, L2Loss, TetwiseL2Loss, TwoModeL2Loss, TwoModeTetwiseL2Loss
from src.minimizer import Minimizer, AdamTorch, AdamManual

class Experiment(ABC):

    def __init__(self,
                exp_file: List[str], #list of len 1
                initial_parameters: List[float],
                perturb_type: str,
                optimizable: List[bool],
                num_iters: int,
                lr: float,
                background_threshold: List[float], #list of len 1
                view: List[int] = [-1], #list of len 1
                device: str = 'cuda',
                set_seed: bool = False):
        self.descriptor = Descriptor(exp_file[0], device, set_seed)
        self.initial_parameters = initial_parameters
        self.optimizable = optimizable
        self.num_iters = num_iters
        self.lr = lr
        self.perturb_type = perturb_type
        self.background_threshold = background_threshold[0]
        self.view = view[0]
    
    @abstractproperty
    def parameters(self) -> Parameters:
        pass

    @abstractproperty
    def simulation(self) -> Simulation:
        pass

    @abstractproperty
    def render(self) -> Render:
        pass

    @abstractproperty
    def camera(self) -> Camera:
        pass

    @abstractproperty
    def loss(self) -> Loss:
        pass
    
    @abstractproperty
    def minimizer(self) -> Minimizer:
        pass

    def run(self) -> None:
        self.minimizer.minimize()


class Static(Experiment):

    @cached_property
    def render(self):
        return StaticRender(self.descriptor, self.simulation, self.camera)

    @cached_property
    def camera(self):
        return StaticCamera(self.descriptor, self.background_threshold, self.view)


class Dynamic(Experiment):
    
    @cached_property
    def render(self):
        return DynamicRender(self.descriptor, self.simulation, self.camera)

    @cached_property
    def camera(self):
        return DynamicCamera(self.descriptor, self.background_threshold)


class NonTetwise(Experiment):

    @cached_property
    def parameters(self):
        return Parameters(self.descriptor, self.initial_parameters, self.optimizable, self.perturb_type)

    @cached_property
    def loss(self):
        return L2Loss(self.descriptor, self.parameters, self.render, self.camera)

    @cached_property
    def minimizer(self):
        return AdamManual(self.loss, self.num_iters, self.lr)


class Tetwise(Experiment):

    @cached_property
    def parameters(self):
        return TetwiseParameters(self.descriptor, self.initial_parameters, self.optimizable, self.perturb_type)

    @cached_property
    def loss(self):
        return TetwiseL2Loss(self.descriptor, self.parameters, self.render, self.camera)

    @cached_property
    def minimizer(self):
        return AdamManual(self.loss, self.num_iters, self.lr)


class StaticBendExp(Static, NonTetwise):

    @cached_property
    def simulation(self):
        return StaticBendSimulation(self.descriptor, self.parameters)


class StaticBendTetwiseExp(Tetwise, StaticBendExp):
    pass


class StaticTwistExp(Static, NonTetwise):

    @cached_property
    def simulation(self):
        return StaticTwistSimulation(self.descriptor, self.parameters)


class StaticTwistTetwiseExp(Tetwise, StaticTwistExp):
    pass


class DynamicBendExp(Dynamic, NonTetwise):

    @cached_property
    def simulation(self):
        return DynamicBendSimulation(self.descriptor, self.parameters)

class DynamicTwistExp(Dynamic, NonTetwise):

    @cached_property
    def simulation(self):
        return DynamicTwistSimulation(self.descriptor, self.parameters)

class BendTwistExperiment(Experiment):

    def __init__(self,
                exp_file: List[str], # [bend_exp_file, twist_exp_file]
                initial_parameters: List[float],
                perturb_type: str,
                optimizable: List[bool],
                num_iters: int,
                lr: float,
                background_threshold: List[float], # [bend_bg_thr, twist_bg_thr]
                view: List[int] = [-1,-1], # [bend_view, twist_view]
                device: str = 'cuda',
                set_seed: bool = False,
                alpha: float = 0.5):

        self.descriptor = Descriptor(exp_file[0], device, set_seed), Descriptor(exp_file[1], device, set_seed)
        self.initial_parameters = initial_parameters
        self.optimizable = optimizable
        self.num_iters = num_iters
        self.lr = lr
        self.background_threshold = background_threshold
        self.view = view
        self.perturb_type = perturb_type
        self.alpha = alpha

class StaticBendTwist(BendTwistExperiment):

    @cached_property
    def render(self):
        return StaticRender(self.descriptor[0], self.simulation[0], self.camera[0]),\
                StaticRender(self.descriptor[1], self.simulation[1], self.camera[1])

    @cached_property
    def camera(self):
        return StaticCamera(self.descriptor[0], self.background_threshold[0], self.view[0]),\
                StaticCamera(self.descriptor[1], self.background_threshold[1], self.view[1])

    @cached_property
    def simulation(self):
        return StaticBendSimulation(self.descriptor[0], self.parameters), \
                StaticTwistSimulation(self.descriptor[1], self.parameters)

class DynamicBendTwist(BendTwistExperiment):

    @cached_property
    def render(self):
        return DynamicRender(self.descriptor[0], self.simulation[0], self.camera[0]),\
                DynamicRender(self.descriptor[1], self.simulation[1], self.camera[1])

    @cached_property
    def camera(self):
        return DynamicCamera(self.descriptor[0], self.background_threshold[0]),\
                DynamicCamera(self.descriptor[1], self.background_threshold[1])

    @cached_property
    def simulation(self):
        return DynamicBendSimulation(self.descriptor[0], self.parameters), \
                DynamicTwistSimulation(self.descriptor[1], self.parameters)

class NonTetwiseBendTwist(BendTwistExperiment):

    @cached_property
    def parameters(self):
        return Parameters(self.descriptor[0], self.initial_parameters, self.optimizable, self.perturb_type)

    @cached_property
    def loss(self):
        return TwoModeL2Loss(self.parameters, self.descriptor[0], self.render[0], self.camera[0],
                                                self.descriptor[1], self.render[1], self.camera[1], self.alpha)

    @cached_property
    def minimizer(self):
        return AdamManual(self.loss, self.num_iters, self.lr)

class TetwiseBendTwist(BendTwistExperiment):

    @cached_property
    def parameters(self):
        return TetwiseParameters(self.descriptor[0], self.initial_parameters, self.optimizable, self.perturb_type)

    @cached_property
    def loss(self):
        return TwoModeTetwiseL2Loss(self.parameters, self.descriptor[0], self.render[0], self.camera[0],
                                                self.descriptor[1], self.render[1], self.camera[1], self.alpha)

    @cached_property
    def minimizer(self):
        return AdamManual(self.loss, self.num_iters, self.lr)


class StaticBendTwistExp(StaticBendTwist, NonTetwiseBendTwist):
    pass

class StaticBendTwistTetwiseExp(StaticBendTwist, TetwiseBendTwist):
    pass

class DynamicBendTwistExp(DynamicBendTwist, NonTetwiseBendTwist):
    pass

class DynamicBendTwistTetwiseExp(DynamicBendTwist, TetwiseBendTwist):
<<<<<<< HEAD
    pass



=======
    pass
>>>>>>> 299ccbacc12c72009bfc61ef8be5ec8f4a581eb6
