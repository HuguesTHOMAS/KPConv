
![Intro figure](https://github.com/HuguesTHOMAS/KPConv/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently not supported as the code uses tensorflow custom operations.


## Experiments

We provide scripts for many experiments:

* Training on a model classification task (ModelNet40)
* Training on a model segmentation task (ShapeNetPart)
* Training on point cloud datasets (S3DIS, Scannet, Semantic3D, NPM3D)
* Test of any of the models
* Visualization of learned features
* Visualization of learned kernel deformations
* Visualization of Effective Receptive Fields

You will find more details on how to run these experiments and how to train a KPConv network on your own data in the [INSTALL.md](./doc) folder

## License
Our code is released under Apache 2.0 License (see LICENSE file for details).

## Updates
* 17/03/2019: New version of KPConv.
* 11/12/2018: Added general visualization code.
* 10/12/2018: Added training code for S3DIS and general test code.
* 26/11/2018: Added training code for ModelNet40/ShapeNetPart.

