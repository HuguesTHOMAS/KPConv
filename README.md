
![Intro figure](https://github.com/HuguesTHOMAS/KPConv/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

### Update

**This is an alpha version of the code**, more features will be added in the next weeks.

### Paper

[arXiv](https://arxiv.org/abs/1904.08889)
```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {arXiv preprint arXiv:1904.08889},
    Year = {2019}
}
```

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently not supported as the code uses tensorflow custom operations.


## Experiments

We provide scripts for many experiments. The instructions to run these experiments are in the [doc](./doc) folder.

* [Object Classification](./doc/object_classification_guide.md): Instructions to train KP-CNN on an object classification
 task (Modelnet40).
 
* [Object Segmentation](./doc/object_segmentation_guide.md): Instructions to train KP-FCNN on an object segmentation task
 (ShapeNetPart)
 
* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on several scene segmentation 
 tasks (S3DIS, Scannet, Semantic3D, NPM3D).
 
* Training KPConv networks on your own data (TODO).

* Test of any of the models (TODO).

* Plot many values (time, loss, validation accuracies, etc.) for any training log (TODO).

* Visualization of learned features (TODO).

* Visualization of learned kernel deformations (TODO).

* Visualization of Effective Receptive Fields (TODO).


## Acknowledgment

Our code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License
Our code is released under MIT License (see LICENSE file for details).

## Updates
* 19/04/2019: Initial release.

