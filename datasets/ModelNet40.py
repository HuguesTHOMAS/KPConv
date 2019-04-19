#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling ModelNet40 dataset
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

# Basic libs
import tensorflow as tf
import numpy as np
import time
import pickle
import os

# PLY and OFF reader
from utils.ply import read_ply


# OS functions
from os import listdir
from os.path import exists, join

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class ModelNet40Dataset(Dataset):
    """
    Class to handle the subset of Modelnet 40 dataset for the mini challenge
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8):
        Dataset.__init__(self, 'ModelNet40')

        ###########################
        # Object classes parameters
        ###########################

        # Dict from labels to names
        self.label_to_names = {0: 'airplane',
                               1: 'bathtub',
                               2: 'bed',
                               3: 'bench',
                               4: 'bookshelf',
                               5: 'bottle',
                               6: 'bowl',
                               7: 'car',
                               8: 'chair',
                               9: 'cone',
                               10: 'cup',
                               11: 'curtain',
                               12: 'desk',
                               13: 'door',
                               14: 'dresser',
                               15: 'flower_pot',
                               16: 'glass_box',
                               17: 'guitar',
                               18: 'keyboard',
                               19: 'lamp',
                               20: 'laptop',
                               21: 'mantel',
                               22: 'monitor',
                               23: 'night_stand',
                               24: 'person',
                               25: 'piano',
                               26: 'plant',
                               27: 'radio',
                               28: 'range_hood',
                               29: 'sink',
                               30: 'sofa',
                               31: 'stairs',
                               32: 'stool',
                               33: 'table',
                               34: 'tent',
                               35: 'toilet',
                               36: 'tv_stand',
                               37: 'vase',
                               38: 'wardrobe',
                               39: 'xbox'}
        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        self.network_model = 'classification'

        # Number of input threads
        self.num_threads = input_threads

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.path = 'Data/ModelNet40'
        self.data_folder = 'modelnet40_normal_resampled'

        # Number of models
        self.num_train = 9843
        self.num_test = 2468

        # Number of thread for input pipeline
        self.num_threads = input_threads

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Subsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_normals = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        ################
        # Training files
        ################

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading training points')
        filename = join(self.path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.input_points['training'], \
                self.input_normals['training'], \
                self.input_labels['training'] = pickle.load(file)

        # Else compute them from original points
        else:

            # Collect training file names
            names = np.loadtxt(join(self.path, self.data_folder, 'modelnet40_train.txt'), dtype=np.str)

            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = join(self.path, self.data_folder, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if subsampling_parameter > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['training'] += [points]
                self.input_normals['training'] += [normals]

            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['training'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['training'],
                             self.input_normals['training'],
                             self.input_labels['training']), file)

        lengths = [p.shape[0] for p in self.input_points['training']]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        ############
        # Test files
        ############

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading test points')
        filename = join(self.path, 'test_{:.3f}_record.pkl'.format(subsampling_parameter))
        if exists(filename):
            with open(filename, 'rb') as file:
                self.input_points['validation'], \
                self.input_normals['validation'], \
                self.input_labels['validation'] = pickle.load(file)

        # Else compute them from original points
        else:

            # Collect test file names
            names = np.loadtxt(join(self.path, self.data_folder, 'modelnet40_test.txt'), dtype=np.str)

            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = join(self.path, self.data_folder, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if subsampling_parameter > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['validation'] += [points]
                self.input_normals['validation'] += [normals]


            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['validation'] = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['validation'],
                             self.input_normals['validation'],
                             self.input_labels['validation']), file)

        lengths = [p.shape[0] for p in self.input_points['validation']]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s\n'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        small = False
        if small:

            for split in ['training', 'validation']:

                pick_n = 50
                gen_indices = []
                for l in self.label_values:
                    label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                    if len(label_inds) > pick_n:
                        label_inds = label_inds[:pick_n]
                    gen_indices += [label_inds.astype(np.int32)]
                gen_indices = np.hstack(gen_indices)

                self.input_points[split] = np.array(self.input_points[split])[gen_indices]
                self.input_normals[split] = np.array(self.input_normals[split])[gen_indices]
                self.input_labels[split] = np.array(self.input_labels[split])[gen_indices]

                if split == 'training':
                    self.num_train = len(gen_indices)
                else:
                    self.num_test = len(gen_indices)

        # Test = validation
        self.input_points['test'] = self.input_points['validation']
        self.input_normals['test'] = self.input_normals['validation']
        self.input_labels['test'] = self.input_labels['validation']

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        # Balance training sample classes
        balanced = False

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}

        # Reset potentials
        self.potentials[split] = np.random.rand(len(self.input_labels[split])) * 1e-3

        ################
        # Def generators
        ################

        def random_balanced_gen():

            # Initiate concatenation lists
            tp_list = []
            tn_list = []
            tl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                if balanced:
                    pick_n = int(np.ceil(self.num_train / self.num_classes))
                    gen_indices = []
                    for l in self.label_values:
                        label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                        rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                        gen_indices += [rand_inds]
                    gen_indices = np.random.permutation(np.hstack(gen_indices))
                else:
                    gen_indices = np.random.permutation(self.num_train)

            elif split == 'validation':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            elif split == 'test':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:

                # Get points
                new_points = self.input_points[split][p_i].astype(np.float32)
                new_normals = self.input_normals[split][p_i].astype(np.float32)
                n = new_points.shape[0]

                # Collect labels
                input_label = self.label_to_idx[self.input_labels[split][p_i]]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tn_list, axis=0),
                           np.array(tl_list, dtype=np.int32),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tn_list = []
                    tl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                tn_list += [new_normals]
                tl_list += [input_label]
                ti_list += [p_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tn_list, axis=0),
                   np.array(tl_list, dtype=np.int32),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(stacked_points, stacked_normals, labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 3:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_normals), axis=1)
            elif config.in_features_dim == 5:
                angles = tf.asin(tf.abs(stacked_normals)) * (2 / np.pi)
                stacked_features = tf.concat((stacked_features, angles), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, stacked_normals), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_classification_inputs(config,
                                                       stacked_points,
                                                       stacked_features,
                                                       labels,
                                                       stack_lengths,
                                                       batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        return tf_map

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        n_b = config.batch_num
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-6]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d} => b = {:.1f}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1],
                                         n_b))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_batches(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        mean_b = 0
        min_b = 1000000
        max_b = 0
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-8]

                max_ind = np.max(batches)
                batches_len = [np.sum(b < max_ind-0.5) for b in batches]

                for b_l in batches_len:
                    mean_b = 0.99 * mean_b + 0.01 * b_l
                max_b = max(max_b, np.max(batches_len))
                min_b = min(min_b, np.min(batches_len))

                print('{:d} < {:.1f} < {:d} /'.format(min_b, mean_b, max_b),
                      self.training_batch_limit,
                      batches_len)

                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_neighbors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        hist_n = 500
        neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                for neighb_mat in neighbors:
                    print(neighb_mat.shape)

                counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                hists = [np.bincount(c, minlength=hist_n) for c in counts]

                neighb_hists += np.vstack(hists)

                print('***********************')
                dispstr = ''
                fmt_l = len(str(int(np.max(neighb_hists)))) + 1
                for neighb_hist in neighb_hists:
                    for v in neighb_hist:
                        dispstr += '{num:{fill}{width}}'.format(num=v, fill=' ', width=fmt_l)
                    dispstr += '\n'
                print(dispstr)
                print('***********************')

                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) +1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 3 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        labels = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(labels)) / np.prod(labels.shape)
        print('object_labels =>', labels.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings nums :\n')

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))


        print('\nFinished\n\n')
        time.sleep(0.5)

        self.flat_inputs = [tf.Variable(in_np, trainable=False) for in_np in inputs]





