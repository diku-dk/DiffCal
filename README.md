# Differentiable Depth for Real2Sim Calibration of Soft Body Simulations

This repository contains the code and the data to calibrate material model parameters for soft body simulation. We use two types of materials i.e. Ecoflex-50 (red) and MoldStar-15 (blue). We have also experimented with heterogenous objects. The shapes include cantilever, spine, XYZ RGB Dragon, and robotic finger. Deformation modes are free hanging, twisting, and oscillating.

![objects](https://user-images.githubusercontent.com/101255383/166220869-53fef7a8-ca4e-45da-baf5-c3454f73819b.png)

## Requirements
All python dependencies are listed in `requirements.txt`. To create a virtual environment and install the required packages run
```
./scripts/dependencies.sh
```
## Folder Structure
* `CAD_models` has the digital model of the all the object shapes in our experiments.
* `exp_data` has the data and the results of all the experiments in the paper and more. `exp_data/*/*.exp` has the configuration of each experiment. Please read the `exp_data/naming_guide.txt` for the subfolder naming convention. In `exp_data/*/camera_data`, you can find the LIDAR depth images as `d_*.npy`, the RGB images`c_*.npy`, and the view transform matrix as `v_*.npy`. You can find the results of the 10 experiment runs with random initial parameters in `exp_data/*/results/log'. They contain loss, gradient, and the parameters value over training iterations.
* 
## Method
Every part in our pipeline is differentiable, as a result we can use the chain rule to get the gradient of the scalar loss function w.r.t the model parameters. The simulator is a function of the parameters and outputs the state which is then rendered into an image. We then minimize the L2 norm between the rendered images and the target depth images, obtained by a LIDAR camera. The code for each segment can be found under `src` directory.

![methods_overview](https://user-images.githubusercontent.com/101255383/166422034-8600be39-0992-4bff-a8fa-30ed4a9e22c7.png)

## Free Hanging Experiment
To run this experiment, use the following bash scripts depending on the type of the material and the shape of the object.
```
./scripts/static_hang_ecoflex_cantilever.sh
./scripts/static_hang_moldstar_cantilever.sh
./scripts/static_hang_ecoflex_spine.sh
./scripts/static_hang_moldstar_spine.sh
./scripts/static_hang_ecoflex+moldstar_cantilever.sh
```
## Twisting Experiment
To run this experiment, use the following.
```
./scripts/static_twist_ecoflex+moldstar_cantilever.sh
```

## Tetwise Experiment
To run this experiment in hanging deformation mode, run
```
./scripts/static_hang_ecoflex+moldstar_cantilever_tetwise.sh
```
To run this experiment in twisting mode, run
```
./scripts/static_twist_ecoflex+moldstar_cantilever_tetwise.sh
```
You can also share the same parameters and optimize for both modes simultaneously.
```
./scripts/static_hang+twist_ecoflex+moldstar_cantilever_tetwise.sh
```

## Viscosity Estimation
```
./scripts/dynamic_hang_moldstar_cantilever.sh
```



https://user-images.githubusercontent.com/101255383/167379976-bce8bd37-67c9-42bf-b191-37ace72f6950.mp4

