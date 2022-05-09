# Differentiable Depth for Real2Sim Calibration of Soft Body Simulations
This repository contains the code and the data to calibrate material model parameters for soft body simulation. We use two types of materials i.e. Ecoflex-50 (red) and MoldStar-15 (blue). We have also experimented with heterogenous objects. The shapes include cantilever, spine, XYZ RGB Dragon, and robotic finger. Deformation modes are free hanging, twisting, and oscillating.
![objects](https://user-images.githubusercontent.com/101255383/166220869-53fef7a8-ca4e-45da-baf5-c3454f73819b.png)

## Requirements
All python dependencies are listed in `requirements.txt`. To create a virtual environment and install the required packages run
```
./scripts/dependencies.sh
```

## Method
Every part in our pipeline is differentiable, as a result we can use the chain rule to get the gradient of the scalar loss function w.r.t the model parameters. The simulator is a function of the parameters and outputs the state which is then rendered into an image. We then minimize the L2 norm between the rendered images and the target depth images, obtained by a LIDAR camera.
![methods_overview](https://user-images.githubusercontent.com/101255383/166422034-8600be39-0992-4bff-a8fa-30ed4a9e22c7.png)

## Folder Structure
* `CAD_models` has the digital model of the all the object shapes in our experiments.
* `exp_data` has the data and the results of all the experiments in the paper and more.
  * `exp_data/*/*.exp` has the configuration of each experiment. Please read the `exp_data/naming_guide.txt` for the subfolder naming convention.
  * In `exp_data/*/camera_data`, you can find the LIDAR depth images as `d_*.npy`, the RGB images`c_*.npy`, and the view transform matrix as `v_*.npy`.
  * You can find the results of the 10 experiment runs with random initial parameters in `exp_data/*/results/log`. They contain loss, gradient, and the parameters value over training iterations.
* `src` has the code for each part introduced in Method section. 
* `main.py` runs the specified experiment. To get help with the experiment configuration, use `python main.py --help`.

## Experiments
The following table summerizes the experiments conducted.
![Untitled (1)](https://user-images.githubusercontent.com/101255383/167411542-bcc31469-59c2-4674-b959-79680579b55c.png)

## Free Hanging Experiment
To run this experiment, use the following bash scripts depending on the type of the material and the shape of the object.
```
./scripts/static_hang_ecoflex_cantilever.sh
./scripts/static_hang_ecoflex_spine.sh
./scripts/static_hang_moldstar_cantilever.sh
./scripts/static_hang_moldstar_spine.sh
./scripts/static_hang_ecoflex+moldstar_cantilever.sh
./scripts/static_hang_ecoflex_dragon.sh
```
![Untitled](https://user-images.githubusercontent.com/101255383/167410437-bd36f7fa-b25e-4964-91ab-bdeadc9f356c.png)

## Twisting Experiment
To run this experiment, use the following.
```
./scripts/static_twist_ecoflex+moldstar_cantilever.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411063-49ef2b98-25fe-425b-a654-5a95c0d0c3a8.png)

## Tetwise Experiment
To run this experiment in hanging deformation mode, run
```
./scripts/static_hang_ecoflex+moldstar_cantilever_tetwise.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411148-283f65cb-6f7b-425e-b688-d935f0e100f3.png)
To run this experiment in twisting mode, run
```
./scripts/static_twist_ecoflex+moldstar_cantilever_tetwise.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411213-c5824684-523a-4fac-a98c-b68c0db43a6b.png)
You can also share the same parameters and optimize for both modes simultaneously.
```
./scripts/static_hang+twist_ecoflex+moldstar_cantilever_tetwise.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411255-1987df9a-de10-400e-8a45-8d7fbdd21ebd.png)

## Viscosity Estimation
```
./scripts/dynamic_hang_moldstar_cantilever.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411341-e00dd48e-f8c3-4d40-9623-e1e577f48832.png)

## Making Soft Robots
In the video below, the process of creating a simple cable-driven robot can be found.
For a simple robot design, the manufacturing pipeline can be divided into the following steps:
- Designing a 3D model of the robot (we have been using OpenSCAD with great success)
- Creating a mold from the model from (1)
- Slicing the 3D model of the mold (using a tool such as Cura)
- Printing the mold on a 3D printer (we currently have 4 x Creality CR-10 Max printers and some smaller + older printers)
- Mixing silicone (Note that the silicone we use is a 2-part solution and that it starts curing as soon as the two parts are mixed)
- Removing air bubbles by placing the solution in a vacuum chamber
- Placing any internal structures in the mold (Optional)
- Pouring in silicone
- Removing the robot from the mold
https://user-images.githubusercontent.com/101255383/167379976-bce8bd37-67c9-42bf-b191-37ace72f6950.mp4

