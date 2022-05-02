# Differentiable Depth for Real2Sim Calibration of Soft Body Simulations

This repository contains the code and the data to estimate elastic parameters of soft bodies. We use two types of materials i.e. Ecoflex-50 (red) and MoldStar-15 (blue). We have also experimented with heterogenous objects. The shapes include cantilever, spine, XYZ RGB Dragon, and robotic finger. Deformation modes are free hanging, twisting, and oscillating.

![objects](https://user-images.githubusercontent.com/101255383/166220869-53fef7a8-ca4e-45da-baf5-c3454f73819b.png)

## 0. Requirements
All python dependencies are listed in `requirements.txt`. To create a virtual environment and install the required packages run
```
./scripts/dependencies.sh
```
