# Differentiable Depth for Real2Sim Calibration of Soft Body Simulations
This repository contains the code and the data to calibrate material model parameters for soft body simulation. We use two types of materials, i.e. Ecoflex-50 (red) and MoldStar-15 (blue). We have also experimented with heterogeneous objects. The shapes include cantilever, spine, XYZ RGB Dragon, and robotic finger. Deformation modes are free hanging, twisting, and oscillating.
![objects](https://user-images.githubusercontent.com/101255383/166220869-53fef7a8-ca4e-45da-baf5-c3454f73819b.png)

## Requirements
All python dependencies are listed in `requirements.txt`. To create a virtual environment and install the required packages run
```
./scripts/dependencies.sh
```
The dFlex package used in `src/simulation.py` can be obtained after having agreed to EULA. 

## Method
Every part in our pipeline is differentiable, as a result we can use the chain rule to get the gradient of the scalar loss function w.r.t the model parameters. The simulator is a function of the parameters and outputs the state which is then rendered into an image. We then minimize the L2 norm between the rendered images and the target depth images, obtained by a LIDAR camera.

<p align="center">
<img src="https://user-images.githubusercontent.com/101255383/166422034-8600be39-0992-4bff-a8fa-30ed4a9e22c7.png" width=750>
</p>

## Folder Structure
* `CAD_models` has the digital model of the all the object shapes in our experiments.
* `exp_data` has the data and the results of all the experiments in the paper and more.
   * `exp_data/*/*.exp` has the configuration of each experiment. Please read the `exp_data/naming_guide.txt` for the subfolder naming convention.
   * In `exp_data/*/camera_data`, you can find the LIDAR depth images as `d_*.npy`, the RGB images`c_*.npy`, and the view transform matrix as `v_*.npy`.
   * You can find the results of the 10 experiment runs with random initial parameters in `exp_data/*/results/log`. They contain loss, gradient, and the parameters value over training iterations.
* `src` has the code for each part introduced in Method section. 
* `main.py` runs the specified experiment. To get help with the experiment configuration, use `python main.py --help`.

## Experiments
The following table summerizes the experiments conducted. The goal is to find the parameter(s) such that the rendered image resembles the depth camera image. The experiments are repeated with 10 random initial parameters to study robustness. In our experiments, the Possion ratio is kept fixed, but one could also optimize with respect to it as well.

![Untitled (1)](https://user-images.githubusercontent.com/101255383/167411542-bcc31469-59c2-4674-b959-79680579b55c.png)

## Passive Deformation
The object is allowed to hang freely under gravity and depth images are obtained in steady state.

To run this experiment, use the following bash scripts depending on the type of the material and the shape of the object.
```
./scripts/static_hang_ecoflex_cantilever.sh
./scripts/static_hang_ecoflex_spine.sh
./scripts/static_hang_moldstar_cantilever.sh
./scripts/static_hang_moldstar_spine.sh
```
<p align="center">
<img src="https://user-images.githubusercontent.com/101255383/167427922-c9ade1ab-ec4e-4107-ae76-f882c4659769.png" width=500>
  </p>

The following figure shows the rendered beam in gray overlayed on depth image camera in color. The simulation matches the real world when optimized.

<p align="center">
<img src="https://user-images.githubusercontent.com/101255383/167427841-8223aca7-4bde-4b0e-aecd-934f17a1321f.png" width=500>
  </p>

## Heterogeneous Materials
We have fabricated objects made of two types of silicone. Each material is assigned its own Young's modulus which is initially assumed to have the same random value.
Other than hanging, we have also studied these objects when they are actively twisted.

To run this experiment, use the following.
```
./scripts/static_hang_ecoflex+moldstar_cantilever.sh
./scripts/static_twist_ecoflex+moldstar_cantilever.sh
```
<p align="center">
<img src="https://user-images.githubusercontent.com/101255383/167428088-4fa8ec47-9d71-4f93-bc0e-16c5df28a373.png" width=500>
  </p>

## Tetwise Experiment
Objects might not be perfectly uniform in their material properties. Thus, we assume each tetrahedron has its own Young's modulus which can be optimized. It is more interesting to study this assumption for objects made of two types of material. We can then visualize the spatial distribution of Young's moduli. We have experimented with hanging and twisting deformation modes separatly and jointly. In the joint experiment, both modes are minimized with equal weighting while sharing the same parameters. 

To run this experiment in hanging or twisting modes, run
```
./scripts/static_hang_ecoflex+moldstar_cantilever_tetwise.sh
./scripts/static_twist_ecoflex+moldstar_cantilever_tetwise.sh
./scripts/static_hang+twist_ecoflex+moldstar_cantilever_tetwise.sh
```
![123](https://user-images.githubusercontent.com/101255383/167430176-83bacd20-6c59-4c73-b77c-edcbf1f2ec10.png)

## Complex Shapes
Our method also deals well with objects with complex shapes. We have experimented with XYZ RBG dragon, which can be run by
```
./scripts/static_hang_ecoflex_dragon.sh
```
<p align="center">
<img src="https://user-images.githubusercontent.com/101255383/167430850-13c7fbae-4981-40fe-aa85-30b828ed3d1b.png" width=500>
  </p>

## Viscosity Estimation
We can also use our method to calibrate the dynamic motion of soft bodies. In addition to the Young's modulus, the density of the object and the damping factor are also optimized in this experiment. In this setup, depth images are captured continously while the beam is oscillating. The following runs this experiment:
```
./scripts/dynamic_hang_moldstar_cantilever.sh
```
![image](https://user-images.githubusercontent.com/101255383/167411341-e00dd48e-f8c3-4d40-9623-e1e577f48832.png)

## Making Soft Robots
In the video below, the process of creating a simple cable-driven robot can be found.
For a simple robot design, the manufacturing pipeline can be divided into the following steps:
1. Designing a 3D model of the robot (we have been using OpenSCAD with great success)
2. Creating a mold from the model from (1)
3. Slicing the 3D model of the mold (using a tool such as Cura)
4. Printing the mold on a 3D printer (we currently have 4 x Creality CR-10 Max printers and some smaller + older printers)
5. Mixing silicone (Note that the silicone we use is a 2-part solution and that it starts curing as soon as the two parts are mixed)
6. Removing air bubbles by placing the solution in a vacuum chamber
7. Placing any internal structures in the mold (Optional)
8. Pouring in silicone
9. Removing the robot from the mold

https://user-images.githubusercontent.com/101255383/167379976-bce8bd37-67c9-42bf-b191-37ace72f6950.mp4

## Citation
Please refer to this work using
<p align="center">
K. Arnavaz et al. "Differentiable Depth for Real2Sim Calibration of Soft Body Simulations", Journal of X, 20YY.
  </p>
