#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.visualizer import ModelVisualizer
from models.KPCNN_model import KernelPointCNN
from models.KPFCNN_model import KernelPointFCNN

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset
from datasets.S3DIS import S3DISDataset
from datasets.Semantic3D import Semantic3DDataset
from datasets.NPM3D import NPM3DDataset
from datasets.Scannet import ScannetDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def visu_caller(path, step_ind, relu_idx):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # No augmentation to avoid random inputs
    config.augment_scale_anisotropic = False
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'none'
    config.augment_scale_min = 1.0
    config.augment_scale_max = 1.0
    config.augment_noise = 0.0
    config.augment_occlusion = 'none'
    config.augment_color = 1.0

    config.batch_num = 2
    config.in_radius = 5

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    if config.dataset.startswith('ModelNet40'):
        dataset = ModelNet40Dataset(config.input_threads)
    elif config.dataset == 'S3DIS':
        dataset = S3DISDataset(config.input_threads)
        on_val = True
    elif config.dataset == 'Scannet':
        dataset = ScannetDataset(config.input_threads, load_test=True)
    elif config.dataset.startswith('ShapeNetPart'):
        dataset = ShapeNetPartDataset(config.dataset.split('_')[1], config.input_threads)
    elif config.dataset == 'NPM3D':
        dataset = NPM3DDataset(config.input_threads, load_test=True)
    elif config.dataset == 'Semantic3D':
        dataset = Semantic3DDataset(config.input_threads)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Create subsample clouds of the models
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initiate ERF input pipeleine (only diff is that it is not random)
    dataset.init_ERF_input_pipeline(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    if config.dataset.startswith('ShapeNetPart'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    elif config.dataset.startswith('S3DIS'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    elif config.dataset.startswith('Scannet'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    elif config.dataset.startswith('NPM3D'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    elif config.dataset.startswith('ModelNet40'):
        model = KernelPointCNN(dataset.flat_inputs, config)
    elif config.dataset.startswith('Semantic3D'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    visualizer = ModelVisualizer(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    #####################
    # Start visualization
    #####################

    print('Start visualization')
    print('*******************\n')

    visualizer.show_effective_recep_field(model, dataset, relu_idx)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #
    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_ModelNet40': Automatically retrieve the last trained model on ModelNet40
    #
    #       > 'last_ShapeNetPart': Automatically retrieve the last trained model on ShapeNetPart
    #
    #       > 'last_S3DIS': Automatically retrieve the last trained model on S3DIS
    #
    #       > 'results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    #

    chosen_log = 'results/Log_2019-03-19_19-14-24'  # => S3DIS, rigid KPConv

    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = -1

    #
    #   Eventually you can choose which feature is visualized (index of the deform operation in the network)
    #

    chosen_relu = 20

    #
    #   If you want to modify certain parameters in the Config class, for example, to stop augmenting the input data,
    #   there is a section for it in the function "test_caller" defined above.
    #

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    visu_caller(chosen_log, chosen_snapshot, chosen_relu)



