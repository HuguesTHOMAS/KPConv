
![Intro figure](https://github.com/HuguesTHOMAS/KPConv/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

### Update 27/04/2020: New [PyTorch implementation](https://github.com/HuguesTHOMAS/KPConv-PyTorch) available. With SemanticKitti, and Windows supported.

This repository contains the implementation of **Kernel Point Convolution** (KPConv), a point convolution operator 
presented in our ICCV2019 paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find our work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

**Update 03/05/2019, bug found with TF 1.13 and CUDA 10.** 
We found an internal bug inside tf.matmul operation. It returns absurd values like 1e12, leading to the 
apparition of NaNs in our network. We advise to use the code with CUDA 9.0 and TF 1.12.
More info in [issue #15](https://github.com/HuguesTHOMAS/KPConv/issues/15)

**SemanticKitti Code:** You can download the code used for SemanticKitti submission [here](https://drive.google.com/open?id=12npkHHnqzhhl5i-2q_RD-Cw_urUdWC0J).
It is not clean, has very few explanations, and and could be buggy. Use it only if you are familiar with KPConv
implementation.

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently 
not supported as the code uses tensorflow custom operations.


## Experiments

We provide scripts for many experiments. The instructions to run these experiments are in the [doc](./doc) folder.

* [Object Classification](./doc/object_classification_guide.md): Instructions to train KP-CNN on an object classification
 task (Modelnet40).
 
* [Object Segmentation](./doc/object_segmentation_guide.md): Instructions to train KP-FCNN on an object segmentation task
 (ShapeNetPart)
 
* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on several scene segmentation 
 tasks (S3DIS, Scannet, Semantic3D, NPM3D).
 
* [New Dataset](./doc/new_dataset_guide.md): Instructions to train KPConv networks on your own data.
 
* [Pretrained models](./doc/pretrained_models_guide.md): We provide pretrained weights and instructions to load them.
 
* [Visualization scripts](./doc/visualization_guide.md): Instructions to use the three scripts allowing to visualize: 
the learned features, the kernel deformations and the Effective Receptive Fields.


## Performances

The following tables report the current performances on different tasks and datasets. Some scores have been improved 
since the article submission.

### Classification and segmentation of 3D shapes

| Method | ModelNet40 OA | ShapeNetPart classes mIoU | ShapeNetPart instances mIoU |
| :--- | :---: | :---: | :---: |
| KPConv _rigid_      | **92.9%** | 85.0%   | 86.2%   |
| KPConv _deform_     | 92.7%   | **85.1%** | **86.4%** |

### Segmentation of 3D scenes

| Method | Scannet mIoU |  Sem3D mIoU  |  S3DIS mIoU  |  NPM3D mIoU  |
| :--- | :---: | :---: | :---: | :---: |
| KPConv _rigid_      | **68.6%** | **74.6%** | 65.4%   | 72.3%   |
| KPConv _deform_     | 68.4%   | 73.1%  | **67.1%** | **82.0%** |


## Acknowledgment

Our code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License
Our code is released under MIT License (see LICENSE file for details).

## Updates
* 17/02/2020: Added a link to SemanticKitti code
* 24/01/2020: Bug fixes
* 01/10/2019: Adding visualization scripts.
* 23/09/2019: Adding pretrained models for NPM3D and S3DIS datasets.
* 03/05/2019: Bug found with TF 1.13 and CUDA 10.
* 19/04/2019: Initial release.

