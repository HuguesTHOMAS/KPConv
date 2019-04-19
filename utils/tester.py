#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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
from os import makedirs, listdir
from os.path import exists, join
import time
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix

from tensorflow.python.client import timeline
import json


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_classification(self, model, dataset, num_votes=100):

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate votes
        average_probs = np.zeros((len(dataset.input_labels['test']), nc_model))
        average_counts = np.zeros((len(dataset.input_labels['test']), nc_model))

        mean_dt = np.zeros(2)
        last_display = time.time()
        while np.min(average_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []
            count = 0

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits, model.labels, model.inputs['object_inds'])
                    prob, labels, inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Get probs and labels
                    probs += [prob]
                    targets += [labels]
                    obj_inds += [inds]
                    count += prob.shape[0]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(np.min(average_counts),
                                             100 * count / dataset.num_test,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Average votes
            # *************

            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(dataset.input_labels['test'][obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered)
            average_counts[obj_inds] += 1
            average_probs[obj_inds] += (probs - average_probs[obj_inds]) / (average_counts[obj_inds])

            # Save/Display temporary results
            # ******************************

            test_labels = np.array(dataset.label_values)

            # Compute classification results
            C1 = confusion_matrix(dataset.input_labels['test'],
                                  np.argmax(average_probs, axis=1),
                                  test_labels)

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print('Test Accuracy = {:.1f}%'.format(ACC))

            s = ''
            for cc in C1:
                for c in cc:
                    s += '{:d} '.format(c)
                s += '\n'
            print(s)



            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return

    def test_segmentation(self, model, dataset, num_votes=100, num_saves=10):

        ##################
        # Pre-computations
        ##################

        print('Preparing test structures')
        t1 = time.time()

        # Collect original test file names
        original_path = join(dataset.path, 'test_ply')
        object_name = model.config.dataset.split('_')[1]
        test_names = [f[:-4] for f in listdir(original_path) if f[-4:] == '.ply' and object_name in f]
        test_names = np.sort(test_names)

        original_labels = []
        original_points = []
        projection_inds = []
        for i, cloud_name in enumerate(test_names):

            # Read data in ply file
            data = read_ply(join(original_path, cloud_name + '.ply'))
            points = np.vstack((data['x'], -data['z'], data['y'])).T
            original_labels += [data['label'] - 1]
            original_points += [points]

            # Create tree structure and compute neighbors
            tree = KDTree(dataset.test_points[i])
            projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))

        ##########
        # Initiate
        ##########

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
        else:
            test_path = None

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Initiate result containers
        average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]

        #####################
        # Network predictions
        #####################

        mean_dt = np.zeros(2)
        last_display = time.time()
        for v in range(num_votes):

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            all_predictions = []
            all_labels = []
            all_points = []
            all_scales = []
            all_rots = []

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits,
                           model.labels,
                           model.inputs['in_batches'],
                           model.inputs['points'],
                           model.inputs['augment_scales'],
                           model.inputs['augment_rotations'])
                    preds, labels, batches, points, s, R = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Stack all predictions for each class separately
                    max_ind = np.max(batches)
                    for b_i, b in enumerate(batches):

                        # Eliminate shadow indices
                        b = b[b < max_ind - 0.5]

                        # Get prediction (only for the concerned parts)
                        predictions = preds[b]

                        # Stack all results
                        all_predictions += [predictions]
                        all_labels += [labels[b]]
                        all_points += [points[0][b]]
                        all_scales += [s[b_i]]
                        all_rots += [R[b_i, :, :]]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Vote {:d} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(v,
                                             100 * len(all_predictions) / len(original_labels),
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Project predictions on original point clouds
            # ********************************************

            print('\nGetting test confusions')
            t1 = time.time()

            proj_predictions = []
            Confs = []
            for i, cloud_name in enumerate(test_names):

                # Interpolate prediction from current positions to original points
                proj_predictions += [all_predictions[i][projection_inds[i]]]

                # Average prediction across votes
                average_predictions[i] = average_predictions[i] + (proj_predictions[i] - average_predictions[i]) / (v + 1)

                # Compute confusion matrices
                parts = [j for j in range(proj_predictions[i].shape[1])]
                Confs += [confusion_matrix(original_labels[i], np.argmax(average_predictions[i], axis=1), parts)]

            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Save the best/worst segmentations per class
            # *******************************************

            print('Saving test examples')
            t1 = time.time()

            # Regroup confusions per object class
            Confs = np.stack(Confs)
            IoUs = IoU_from_confusions(Confs)
            mIoUs = np.mean(IoUs, axis=-1)

            # Get X best and worst prediction
            order = np.argsort(mIoUs)
            worst_inds = order[:num_saves]
            best_inds = order[:-num_saves-1:-1]
            worst_IoUs = IoUs[order[:num_saves]]
            best_IoUs = IoUs[order[:-num_saves-1:-1]]

            # Save the names in a file
            obj_path = join(test_path, object_name)
            if not exists(obj_path):
                makedirs(obj_path)
            worst_file = join(obj_path, 'worst_inds.txt')
            best_file = join(obj_path, 'best_inds.txt')
            with open(worst_file, "w") as text_file:
                for w_i, w_IoUs in zip(worst_inds, worst_IoUs):
                    text_file.write('{:d} {:s} :'.format(w_i, test_names[w_i]))
                    for IoU in w_IoUs:
                        text_file.write(' {:.1f}'.format(100*IoU))
                    text_file.write('\n')

            with open(best_file, "w") as text_file:
                for b_i, b_IoUs in zip(best_inds, best_IoUs):
                    text_file.write('{:d} {:s} :'.format(b_i, test_names[b_i]))
                    for IoU in b_IoUs:
                        text_file.write(' {:.1f}'.format(100*IoU))
                    text_file.write('\n')

            # Save the clouds
            for i, w_i in enumerate(worst_inds):
                filename = join(obj_path, 'worst_{:02d}.ply'.format(i+1))
                preds = np.argmax(average_predictions[w_i], axis=1).astype(np.int32)
                write_ply(filename,
                          [original_points[w_i], original_labels[w_i], preds],
                          ['x', 'y', 'z', 'gt', 'pre'])

            for i, b_i in enumerate(best_inds):
                filename = join(obj_path, 'best_{:02d}.ply'.format(i+1))
                preds = np.argmax(average_predictions[b_i], axis=1).astype(np.int32)
                write_ply(filename,
                          [original_points[b_i], original_labels[b_i], preds],
                          ['x', 'y', 'z', 'gt', 'pre'])

            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Display results
            # ***************

            print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
            print('-----|------|--------------------------------------------------------------------------------')

            s = '---- | ---- | '
            for obj in dataset.label_names:
                if obj == object_name:
                    s += '{:5.2f} '.format(100 * np.mean(mIoUs))
                else:
                    s += '---- '
            print(s + '\n')

            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return

    def test_multi_segmentation(self, model, dataset, num_votes=100, num_saves=10):

        ##################
        # Pre-computations
        ##################

        print('Preparing test structures')
        t1 = time.time()

        # Collect original test file names
        original_path = join(dataset.path, 'test_ply')
        test_names = [f[:-4] for f in listdir(original_path) if f[-4:] == '.ply']
        test_names = np.sort(test_names)

        original_labels = []
        original_points = []
        projection_inds = []
        for i, cloud_name in enumerate(test_names):

            # Read data in ply file
            data = read_ply(join(original_path, cloud_name + '.ply'))
            points = np.vstack((data['x'], -data['z'], data['y'])).T
            original_labels += [data['label'] - 1]
            original_points += [points]

            # Create tree structure to compute neighbors
            tree = KDTree(dataset.input_points['test'][i])
            projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))

        ##########
        # Initiate
        ##########

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
        else:
            test_path = None

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Initiate result containers
        average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]

        #####################
        # Network predictions
        #####################

        mean_dt = np.zeros(2)
        last_display = time.time()
        for v in range(num_votes):

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            all_predictions = []
            all_obj_inds = []

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits,
                           model.labels,
                           model.inputs['super_labels'],
                           model.inputs['object_inds'],
                           model.inputs['in_batches'])
                    preds, labels, obj_labels, o_inds, batches = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Stack all predictions for each class separately
                    max_ind = np.max(batches)
                    for b_i, b in enumerate(batches):

                        # Eliminate shadow indices
                        b = b[b < max_ind - 0.5]

                        # Get prediction (only for the concerned parts)
                        obj = obj_labels[b[0]]
                        predictions = preds[b][:, :model.config.num_classes[obj]]

                        # Stack all results
                        all_predictions += [predictions]
                        all_obj_inds += [o_inds[b_i]]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Vote {:d} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(v,
                                             100 * len(all_predictions) / dataset.num_test,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Project predictions on original point clouds
            # ********************************************

            print('\nGetting test confusions')
            t1 = time.time()

            for i, probs in enumerate(all_predictions):

                # Interpolate prediction from current positions to original points
                obj_i = all_obj_inds[i]
                proj_predictions = probs[projection_inds[obj_i]]

                # Average prediction across votes
                average_predictions[obj_i] = average_predictions[obj_i] + \
                                             (proj_predictions - average_predictions[obj_i]) / (v + 1)

            Confs = []
            for obj_i, avg_probs in enumerate(average_predictions):

                # Compute confusion matrices
                parts = [j for j in range(avg_probs.shape[1])]
                Confs += [confusion_matrix(original_labels[obj_i], np.argmax(avg_probs, axis=1), parts)]


            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Save the best/worst segmentations per class
            # *******************************************

            print('Saving test examples')
            t1 = time.time()

            # Regroup confusions per object class
            Confs = np.array(Confs)
            obj_mIoUs = []
            for l in dataset.label_values:

                # Get confusions for this object
                obj_inds = np.where(dataset.input_labels['test'] == l)[0]
                obj_confs = np.stack(Confs[obj_inds])

                # Get IoU
                obj_IoUs = IoU_from_confusions(obj_confs)
                obj_mIoUs += [np.mean(obj_IoUs, axis=-1)]

                # Get X best and worst prediction
                order = np.argsort(obj_mIoUs[-1])
                worst_inds = obj_inds[order[:num_saves]]
                best_inds = obj_inds[order[:-num_saves-1:-1]]
                worst_IoUs = obj_IoUs[order[:num_saves]]
                best_IoUs = obj_IoUs[order[:-num_saves-1:-1]]

                # Save the names in a file
                obj_path = join(test_path, dataset.label_to_names[l])
                if not exists(obj_path):
                    makedirs(obj_path)
                worst_file = join(obj_path, 'worst_inds.txt')
                best_file = join(obj_path, 'best_inds.txt')
                with open(worst_file, "w") as text_file:
                    for w_i, w_IoUs in zip(worst_inds, worst_IoUs):
                        text_file.write('{:d} {:s} :'.format(w_i, test_names[w_i]))
                        for IoU in w_IoUs:
                            text_file.write(' {:.1f}'.format(100*IoU))
                        text_file.write('\n')

                with open(best_file, "w") as text_file:
                    for b_i, b_IoUs in zip(best_inds, best_IoUs):
                        text_file.write('{:d} {:s} :'.format(b_i, test_names[b_i]))
                        for IoU in b_IoUs:
                            text_file.write(' {:.1f}'.format(100*IoU))
                        text_file.write('\n')

                # Save the clouds
                for i, w_i in enumerate(worst_inds):
                    filename = join(obj_path, 'worst_{:02d}.ply'.format(i+1))
                    preds = np.argmax(average_predictions[w_i], axis=1).astype(np.int32)
                    write_ply(filename,
                              [original_points[w_i], original_labels[w_i], preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                for i, b_i in enumerate(best_inds):
                    filename = join(obj_path, 'best_{:02d}.ply'.format(i+1))
                    preds = np.argmax(average_predictions[b_i], axis=1).astype(np.int32)
                    write_ply(filename,
                              [original_points[b_i], original_labels[b_i], preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Display results
            # ***************

            objs_average = [np.mean(mIoUs) for mIoUs in obj_mIoUs]
            instance_average = np.mean(np.hstack(obj_mIoUs))
            class_average = np.mean(objs_average)

            print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
            print('-----|------|--------------------------------------------------------------------------------')

            s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
            for AmIoU in objs_average:
                s += '{:4.1f} '.format(100 * AmIoU)
            print(s + '\n')

            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return

    def test_cloud_segmentation(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.data.shape[0], nc_model), dtype=np.float32) for l in dataset.input_trees['test']]

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops,
                                                                                       {model.dropout_prob: 1.0})
                """
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops,
                                                                                       {model.dropout_prob: 1.0},
                                                                                       options=options,
                                                                                       run_metadata=run_metadata)
                """
                t += [time.time()]

                #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                #many_runs_timeline.update_timeline(chrome_trace)

                if False:
                    many_runs_timeline.save('timeline_merged_%d_runs.json' % i0)
                    a = 1/0

                # Get predictions and labels per instance
                # ***************************************

                # Stack all predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs



                # Average timing
                t += [time.time()]
                #print(batches.shape, stacked_probs.shape, 1000*(t[1] - t[0]), 1000*(t[2] - t[1]))
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['test'])))

                i0 += 1

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['test'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))
                print([np.mean(pots) for pots in dataset.potentials['test']])

                if last_min + 2 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):

                        # Get file
                        points = dataset.load_evaluation_points(file_path)

                        # Reproject probs
                        probs = self.test_probs[i_test][dataset.test_proj[i_test], :]

                        # Insert false columns for ignored labels
                        probs2 = probs.copy()
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.int32)

                        # Project potentials on original points
                        pots = dataset.potentials['test'][i_test][dataset.test_proj[i_test]]

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds, pots],
                                  ['x', 'y', 'z', 'preds', 'pots'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values
                                      if label not in dataset.ignored_labels]
                        write_ply(test_name2,
                                  [points, probs],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save ascii preds
                        if dataset.name.startswith('Semantic3D'):
                            ascii_name = join(test_path, 'predictions', dataset.ascii_files[cloud_name])
                        else:
                            ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(dataset.test_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return

    def test_cloud_segmentation_on_val(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions')):
                makedirs(join(test_path, 'val_predictions'))
            if not exists(join(test_path, 'val_probs')):
                makedirs(join(test_path, 'val_probs'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))

                i0 += 1


            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i_test in range(dataset.num_validation):

                        # Insert false columns for ignored labels
                        probs = self.test_probs[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = dataset.input_labels['validation'][i_test]

                        # Confs
                        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                    if int(np.ceil(new_min)) % 4 == 0:

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        t1 = time.time()
                        files = dataset.train_files
                        i_val = 0
                        proj_probs = []
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:

                                # Reproject probs on the evaluations points
                                probs = self.test_probs[i_val][dataset.validation_proj[i_val], :]
                                proj_probs += [probs]
                                i_val += 1

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i_test in range(dataset.num_validation):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    proj_probs[i_test] = np.insert(proj_probs[i_test], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                            # Confusion
                            targets = dataset.validation_labels[i_test]
                            Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        files = dataset.train_files
                        i_test = 0
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:

                                # Get points
                                points = dataset.load_evaluation_points(file_path)

                                # Get the predicted labels
                                preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                                # Project potentials on original points
                                pots = dataset.potentials['validation'][i_test][dataset.validation_proj[i_test]]

                                # Save plys
                                cloud_name = file_path.split('/')[-1]
                                test_name = join(test_path, 'val_predictions', cloud_name)
                                write_ply(test_name,
                                          [points, preds, pots, dataset.validation_labels[i_test]],
                                          ['x', 'y', 'z', 'preds', 'pots', 'gt'])
                                test_name2 = join(test_path, 'val_probs', cloud_name)
                                prob_names = ['_'.join(dataset.label_to_names[label].split())
                                              for label in dataset.label_values]
                                write_ply(test_name2,
                                          [points, proj_probs[i_test]],
                                          ['x', 'y', 'z'] + prob_names)
                                i_test += 1
                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(dataset.val_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return






























