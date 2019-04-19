
![Intro figure](https://github.com/HuguesTHOMAS/KPConv/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently not supported as the code uses tensorflow custom operations.





## Shape classification on ModelNet40

### Data

Regularly sampled clouds from ModelNet40 dataset can be downloaded <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6 GB)</a>. Uncompress the folder and move it to `Data/ModelNet40/modelnet40_normal_resampled`.

N.B. If you want to place your data anywhere else, you just have to change the variable `self.path` of `ModelNet40Dataset` class (line 142 of the file `datasets/ModelNet40.py`). The same can be done for the other datasets.

### Training a model

Simply run the following script to start the training:

        python3 training_ModelNet40.py
        
This file contains a configuration subclass `ModelNet40Config`, inherited from the general configuration class `Config` defined in `utils/config.py`. The value of every parameter can be modified in the subclass. The first run of this script will precompute structures for the dataset which might take some time.

### Test a model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which model you want to test. Follow them and then run the script :

        python3 test_any_model.py

### Visualizations

#### Show Learned features

For any model, run:

        python3 visualize_features.py
        
More details in the script.
        
#### Plot a logged training

For any model, run:

        python3 plot_convergence.py
        
With this script, you can show the evolution of the training loss, validation accuracy, learning rate. You can also compare different logs. More details in the script.

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

