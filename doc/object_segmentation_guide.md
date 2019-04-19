
![Intro figure](https://github.com/HuguesTHOMAS/KPConv/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently not supported as the code uses tensorflow custom operations.

N.B. If you want to place your data anywhere else, you just have to change the variable `self.path` of `ShapeNetPartDataset` class (in the file `datasets/ShapeNetPart.py`).

## Object Part Segmentation on ShapeNetPart

### Data

ShapeNetPart dataset can be downloaded <a href="https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip">here (635 MB)</a>. Uncompress the folder and move it to `Data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0`.

### Training

Simply run the following script to start the training:

        python3 training_ShapeNetPart.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `ShapeNetPartConfig`, and the first run of this script might take some time to precompute dataset structures.

### Test and visualization

See [ModelNet40 section].

## Scene Segmentation on S3DIS

### Data

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. Download the file named `Stanford3dDataset_v1.2.zip`, uncompress the folder and move it to `Data/S3DIS/Stanford3dDataset_v1.2`.

### Training

Simply run the following script to start the training:

        python3 training_S3DIS.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `S3DISConfig`, and the first run of this script might take some time to precompute dataset structures.

### Test and visualization

See [ModelNet40 section].

## License
Our code is released under Apache 2.0 License (see LICENSE file for details).

## Updates
* 17/03/2019: New version of KPConv.
* 11/12/2018: Added general visualization code.
* 10/12/2018: Added training code for S3DIS and general test code.
* 26/11/2018: Added training code for ModelNet40/ShapeNetPart.

