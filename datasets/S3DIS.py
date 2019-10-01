#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle S3DIS dataset in a class
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
import json
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
import os

# PLY reader
from utils.ply import read_ply, write_ply

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

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


class S3DISDataset(Dataset):
    """
    Class to handle S3DIS dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8):
        Dataset.__init__(self, 'S3DIS')

        ###########################
        # Object classes parameters
        ###########################

        # Dict from labels to names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        self.network_model = 'cloud_segmentation'

        # Number of input threads
        self.num_threads = input_threads

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.path = 'Data/S3DIS/Stanford3dDataset_v1.2'

        # Path of the training files
        self.train_path = 'original_ply'

        # List of files to process
        ply_path = join(self.path, self.train_path)

        # Proportion of validation scenes
        self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
        self.all_splits = [0, 1, 2, 3, 4, 5]
        self.validation_split = 4

        # List of training files
        self.train_files = [join(ply_path, f + '.ply') for f in self.cloud_names]

        ###################
        # Prepare ply files
        ###################

        self.prepare_S3DIS_ply()

    def prepare_S3DIS_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = join(ply_path, cloud_name + '.ply')
            if exists(cloud_file):
                continue

            # Get rooms of the current cloud
            cloud_folder = join(self.path, cloud_name)
            room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if isdir(join(cloud_folder, room))]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                print('Cloud %s - Room %d/%d : %s' % (cloud_name, i+1, len(room_folders), room_folder.split('\\')[-1]))

                for object_name in listdir(join(room_folder, 'Annotations')):

                    if object_name[-4:] == '.txt':

                        # Text file containing point of the object
                        object_file = join(room_folder, 'Annotations', object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split('_')[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ['stairs']:
                            object_class = self.name_to_label['clutter']
                        else:
                            raise ValueError('Unknown object name: ' + str(tmp))

                        # Read object points and colors
                        with open(object_file, 'r') as f:
                            object_data = np.array([[float(x) for x in line.split()] for line in f])

                        # Stack all data
                        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                        object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(cloud_file,
                      (cloud_points, cloud_colors, cloud_classes),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        print('Done in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not exists(tree_path):
            makedirs(tree_path)

        # Initiate containers
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}

        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            if self.all_splits[i] == self.validation_split:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, subsampling_parameter))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, subsampling_parameter))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class']

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        print('\nPreparing reprojection indices for testing')

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        i_val = 0
        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # Get info on this cloud
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.all_splits[i] == self.validation_split:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()

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

        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.epoch_steps * config.batch_num
            random_pick_n = None

        elif split == 'validation':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'test':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'ERF':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = 1000000
            self.batch_limit = 1
            np.random.seed(42)

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        data_split = split
        if split == 'ERF':
            data_split = 'validation'
        for i, tree in enumerate(self.input_trees[data_split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        ##########################
        # Def generators functions
        ##########################

        def get_random_epoch_inds():

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.input_labels[split]):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.label_values):
                    if label not in self.ignored_labels:

                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            return all_epoch_inds

        def random_balanced_gen():

            # First choose the point we are going to look at for this epoch
            # *************************************************************

            # This generator cannot be used on test split
            if split == 'training':
                all_epoch_inds = get_random_epoch_inds()
            elif split == 'validation':
                all_epoch_inds = get_random_epoch_inds()
            else:
                raise ValueError('generator to be defined for test split.')

            # Now create batches
            # ******************

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i, rand_i in enumerate(np.random.permutation(all_epoch_inds.shape[1])):

                cloud_ind = all_epoch_inds[0, rand_i]
                point_ind = all_epoch_inds[1, rand_i]

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.input_trees[split][cloud_ind].query_radius(pick_point,
                                                                             r=config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Safe check for very dense areas
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[split][cloud_ind][input_inds]
                input_labels = self.input_labels[split][cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))

        def spatially_regular_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):


                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                if split != 'ERF':
                    noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                    pick_point = center_point + noise.astype(center_point.dtype)
                else:
                    pick_point = center_point

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                             r=config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                if split != 'ERF':
                    dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(config.in_radius))
                    tukeys[dists > np.square(config.in_radius)] = 0
                    self.potentials[split][cloud_ind][input_inds] += tukeys
                    self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                    # Safe check for very dense areas
                    if n > self.batch_limit:
                        input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                        n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split in ['test', 'ERF']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:

                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))

        ###################
        # Choose generators
        ###################

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen

        elif split == 'validation':
            gen_func = spatially_regular_gen

        elif split in ['test', 'ERF']:
            gen_func = spatially_regular_gen

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            """
            [None, 3], [None, 3], [None], [None]
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            if config.in_features_dim in [4, 5]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 2:
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 3:
                stacked_features = stacked_colors
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif config.in_features_dim == 5:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stacks_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [point_inds, cloud_inds]

            return input_list

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T



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
                batches = np_flat_inputs[-7]
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
                batches = np_flat_inputs[-7]

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
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

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
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_colors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
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
                stacked_points = np_flat_inputs[:config.num_layers]
                stacked_colors = np_flat_inputs[-9]
                batches = np_flat_inputs[-7]
                stacked_labels = np_flat_inputs[-5]

                # Extract a point cloud and its color to save
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get points and colors (only for the concerned parts)
                    points = stacked_points[0][b]
                    colors = stacked_colors[b]
                    labels = stacked_labels[b]

                    write_ply('S3DIS_input_{:d}.ply'.format(b_i),
                              [points, colors[:, 1:4], labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'labels'])

                a = 1/0



                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

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
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) +1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) +1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
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
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')



