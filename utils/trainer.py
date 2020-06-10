#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
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
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import psutil
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Add training ops
        self.add_train_ops(model)

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        """
        print('*************************************')
        sum = 0
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='KernelPointNetwork'):
            #print(var.name, var.shape)
            sum += np.prod(var.shape)
        print('total parameters : ', sum)
        print('*************************************')

        print('*************************************')
        sum = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork'):
            #print(var.name, var.shape)
            sum += np.prod(var.shape)
        print('total parameters : ', sum)
        print('*************************************')
        """

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
            exclude_vars = ['softmax', 'head_unary_conv', '/fc/']
            restore_vars = my_vars
            for exclude_var in exclude_vars:
                restore_vars = [v for v in restore_vars if exclude_var not in v.name]
            restorer = tf.train.Saver(restore_vars)
            restorer.restore(self.sess, restore_snap)
            print("Model restored.")

    def add_train_ops(self, model):
        """
        Add training ops on top of the model
        """

        ##############
        # Training ops
        ##############

        with tf.variable_scope('optimizer'):

            # Learning rate as a Variable so we can modify it
            self.learning_rate = tf.Variable(model.config.learning_rate, trainable=False, name='learning_rate')

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, model.config.momentum)

            # Training step op
            gvs = optimizer.compute_gradients(model.loss)

            if model.config.grad_clip_norm > 0:

                # Get gradient for deformable convolutions and scale them
                scaled_gvs = []
                for grad, var in gvs:
                    if 'offset_conv' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    if 'offset_mlp' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    else:
                        scaled_gvs.append((grad, var))

                # Clipping each gradient independantly
                capped_gvs = [(tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for grad, var in scaled_gvs]

                # Clipping the whole network gradient (problematic with big network where grad == inf)
                # capped_grads, global_norm = tf.clip_by_global_norm([grad for grad, var in gvs], self.config.grad_clip_norm)
                # vars = [var for grad, var in gvs]
                # capped_gvs = [(grad, var) for grad, var in zip(capped_grads, vars)]

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            else:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)

        ############
        # Result ops
        ############

        # Add the Op to compare the logits to the labels during evaluation.
        with tf.variable_scope('results'):

            if len(model.config.ignored_label_inds) > 0:
                #  Boolean mask of points that should be ignored
                ignored_bool = tf.zeros_like(model.labels, dtype=tf.bool)
                for ign_label in model.config.ignored_label_inds:
                    ignored_bool = tf.logical_or(ignored_bool, model.labels == ign_label)

                #  Collect logits and labels that are not ignored
                inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
                new_logits = tf.gather(model.logits, inds, axis=0)
                new_labels = tf.gather(model.labels, inds, axis=0)

                #  Reduce label values in the range of logit shape
                reducing_list = tf.range(model.config.num_classes, dtype=tf.int32)
                inserted_value = tf.zeros((1,), dtype=tf.int32)
                for ign_label in model.config.ignored_label_inds:
                    reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
                new_labels = tf.gather(reducing_list, new_labels)

                # Metrics
                self.correct_prediction = tf.nn.in_top_k(new_logits, new_labels, 1)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                self.prob_logits = tf.nn.softmax(new_logits)

            else:

                # Metrics
                self.correct_prediction = tf.nn.in_top_k(model.logits, model.labels, 1)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                self.prob_logits = tf.nn.softmax(model.logits)

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, model, dataset, debug_NaN=False):
        """
        Train the model on a particular dataset.
        """

        if debug_NaN:
            # Add checking ops
            self.check_op = tf.add_check_numerics_ops()

        # Parameters log file
        if model.config.saving:
            model.parameters_log()

        # Save points of the kernel to file
        self.save_kernel_points(model, 0)

        if model.config.saving:
            # Training log file
            with open(join(model.saving_path, 'training.txt'), "w") as file:
                file.write('Steps out_loss reg_loss point_loss train_accuracy time memory\n')

            # Killing file (simply delete this file when you want to stop the training)
            if not exists(join(model.saving_path, 'running_PID.txt')):
                with open(join(model.saving_path, 'running_PID.txt'), "w") as file:
                    file.write('Launched with PyCharm')

        # Train loop variables
        t0 = time.time()
        self.training_step = 0
        self.training_epoch = 0
        mean_dt = np.zeros(2)
        last_display = t0
        self.training_preds = np.zeros(0)
        self.training_labels = np.zeros(0)
        epoch_n = 1
        mean_epoch_n = 0

        # Initialise iterator with train data
        self.sess.run(dataset.train_init_op)

        # Start loop
        while self.training_epoch < model.config.max_epoch:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.train_op,
                       model.output_loss,
                       model.regularization_loss,
                       model.offsets_loss,
                       model.logits,
                       model.labels,
                       self.accuracy]

                # If NaN appears in a training, use this debug block
                if debug_NaN:
                    all_values = self.sess.run(ops + [self.check_op] + list(dataset.flat_inputs), {model.dropout_prob: 0.5})
                    L_out, L_reg, L_p, probs, labels, acc = all_values[1:7]
                    if np.isnan(L_reg) or np.isnan(L_out):
                        input_values = all_values[8:]
                        self.debug_nan(model, input_values, probs)
                        a = 1/0

                else:
                    # Run normal
                    _, L_out, L_reg, L_p, probs, labels, acc = self.sess.run(ops, {model.dropout_prob: 0.5})

                t += [time.time()]

                # Stack prediction for training confusion
                if model.config.network_model == 'classification':
                    self.training_preds = np.hstack((self.training_preds, np.argmax(probs, axis=1)))
                    self.training_labels = np.hstack((self.training_labels, labels))
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} L_out={:5.3f} L_reg={:5.3f} L_p={:5.3f} Acc={:4.2f} ' \
                              '---{:8.2f} ms/batch (Averaged)'
                    print(message.format(self.training_step,
                                         L_out,
                                         L_reg,
                                         L_p,
                                         acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

                # Log file
                if model.config.saving:
                    process = psutil.Process(os.getpid())
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:.3f} {:.3f} {:.3f} {:.2f} {:.2f} {:.1f}\n'
                        file.write(message.format(self.training_step,
                                                  L_out,
                                                  L_reg,
                                                  L_p,
                                                  acc,
                                                  t[-1] - t0,
                                                  process.memory_info().rss * 1e-6))

                # Check kill signal (running_PID.txt deleted)
                if model.config.saving and not exists(join(model.saving_path, 'running_PID.txt')):
                    break

                if model.config.dataset.startswith('ShapeNetPart') or model.config.dataset.startswith('ModelNet'):
                    if model.config.epoch_steps and epoch_n > model.config.epoch_steps:
                        raise tf.errors.OutOfRangeError(None, None, '')

            except tf.errors.OutOfRangeError:

                # End of train dataset, update average of epoch steps
                mean_epoch_n += (epoch_n - mean_epoch_n) / (self.training_epoch + 1)
                epoch_n = 0
                self.int = int(np.floor(mean_epoch_n))
                model.config.epoch_steps = int(np.floor(mean_epoch_n))
                if model.config.saving:
                    model.parameters_log()

                # Snapshot
                if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:

                    # Tensorflow snapshot
                    snapshot_directory = join(model.saving_path, 'snapshots')
                    if not exists(snapshot_directory):
                        makedirs(snapshot_directory)
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step + 1)

                    # Save points
                    self.save_kernel_points(model, self.training_epoch)

                # Update learning rate
                if self.training_epoch in model.config.lr_decays:
                    op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                               model.config.lr_decays[self.training_epoch]))
                    self.sess.run(op)

                # Increment
                self.training_epoch += 1

                # Validation
                if model.config.network_model == 'classification':
                    self.validation_error(model, dataset)
                elif model.config.network_model == 'segmentation':
                    self.segment_validation_error(model, dataset)
                elif model.config.network_model == 'multi_segmentation':
                    self.multi_validation_error(model, dataset)
                elif model.config.network_model == 'cloud_segmentation':
                    self.cloud_validation_error(model, dataset)
                else:
                    raise ValueError('No validation method implemented for this network type')

                self.training_preds = np.zeros(0)
                self.training_labels = np.zeros(0)

                # Reset iterator on training data
                self.sess.run(dataset.train_init_op)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1/0

            # Increment steps
            self.training_step += 1
            epoch_n += 1

        # Remove File for kill signal
        if exists(join(model.saving_path, 'running_PID.txt')):
            remove(join(model.saving_path, 'running_PID.txt'))
        self.sess.close()

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation_error(self, model, dataset):
        """
        Validation method for classification models
        """

        ##########
        # Initiate
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((len(dataset.input_labels['validation']), nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        mean_dt = np.zeros(2)
        last_display = time.time()
        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits, model.labels, model.inputs['object_inds'])
                prob, labels, inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get probs and labels
                probs += [prob]
                targets += [labels]
                obj_inds += [inds]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1-val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(dataset.label_values)

        # Compute classification results
        C1 = confusion_matrix(targets,
                              np.argmax(probs, axis=1),
                              validation_labels)

        # Compute training confusion
        C2 = confusion_matrix(self.training_labels,
                              self.training_preds,
                              validation_labels)

        # Compute votes confusion
        C3 = confusion_matrix(dataset.input_labels['validation'],
                              np.argmax(self.val_probs, axis=1),
                              validation_labels)


        # Saving (optionnal)
        if model.config.saving:
            print("Save confusions")
            conf_list = [C1, C2, C3]
            file_list = ['val_confs.txt', 'training_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(model.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        train_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C3)) / (np.sum(C3) + 1e-6)
        print('Accuracies : train = {:.1f}% / val = {:.1f}% / vote = {:.1f}%'.format(train_ACC, val_ACC, vote_ACC))

        return C1

    def segment_validation_error(self, model, dataset):
        """
        Validation method for single object segmentation models
        """

        ##########
        # Initiate
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = [np.zeros((len(p_l), nc_model)) for p_l in dataset.input_point_labels['validation']]

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits, model.labels, model.inputs['in_batches'], model.inputs['object_inds'])
                prob, labels, batches, o_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Stack all results
                    probs += [prob[b]]
                    targets += [labels[b]]
                    obj_inds += [o_inds[b_i]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        ###################
        # Voting validation
        ###################

        for o_i, o_probs in zip(obj_inds, probs):
            self.val_probs[o_i] = val_smooth * self.val_probs[o_i] + (1 - val_smooth) * o_probs

        ############
        # Confusions
        ############

        # Confusion matrix for each instance
        n_parts = model.config.num_classes
        Confs = np.zeros((len(probs), n_parts, n_parts), dtype=np.int32)
        for i, (pred, truth) in enumerate(zip(probs, targets)):
            parts = [j for j in range(pred.shape[1])]
            Confs[i, :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)

        # Objects IoU
        IoUs = IoU_from_confusions(Confs)


        # Compute votes confusion
        Confs = np.zeros((len(self.val_probs), n_parts, n_parts), dtype=np.int32)
        for i, (pred, truth) in enumerate(zip(self.val_probs, dataset.input_point_labels['validation'])):
            parts = [j for j in range(pred.shape[1])]
            Confs[i, :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)

        # Objects IoU
        vote_IoUs = IoU_from_confusions(Confs)

        # Saving (optionnal)
        if model.config.saving:

            IoU_list = [IoUs, vote_IoUs]
            file_list = ['val_IoUs.txt', 'vote_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(model.saving_path, IoU_file)

                # Line to write:
                line = ''
                for instance_IoUs in IoUs_to_save:
                    for IoU in instance_IoUs:
                        line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        mIoU2 = 100 * np.mean(vote_IoUs)
        print('{:s} : mIoU = {:.1f}% / vote mIoU = {:.1f}%'.format(model.config.dataset, mIoU, mIoU2))

        return

    def cloud_validation_error(self, model, dataset):
        """
        Validation method for cloud segmentation models
        """

        ##########
        # Initiate
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Do not validate if dataset has no validation cloud
        if dataset.validation_split not in dataset.all_splits:
            return

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Number of classes including ignored labels
        nc_tot = dataset.num_classes

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model)) for l in dataset.input_labels['validation']]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in dataset.label_values:
                if label_value not in dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
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
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                                + (1-val_smooth) * probs

                    # Stack all prediction for this epoch
                    predictions += [probs]
                    targets += [dataset.input_labels['validation'][c_i][inds]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(dataset.label_values):
                if label_value in dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
            if label_value in dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        # Saving (optionnal)
        if model.config.saving:

            # Name of saving file
            test_file = join(model.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(model.config.dataset, mIoU))

        # Save predicted cloud occasionally
        if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:
            val_path = join(model.saving_path, 'val_preds_{:d}'.format(self.training_epoch))
            if not exists(val_path):
                makedirs(val_path)
            files = dataset.train_files
            i_val = 0
            for i, file_path in enumerate(files):
                if dataset.all_splits[i] == dataset.validation_split:

                    # Get points
                    points = dataset.load_evaluation_points(file_path)

                    # Get probs on our own ply points
                    sub_probs = self.validation_probs[i_val]

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(dataset.label_values):
                        if label_value in dataset.ignored_labels:
                            sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                    # Get the predicted labels
                    sub_preds = dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                    # Reproject preds on the evaluations points
                    preds = (sub_preds[dataset.validation_proj[i_val]]).astype(np.int32)

                    # Path of saved validation file
                    cloud_name = file_path.split('/')[-1]
                    val_name = join(val_path, cloud_name)

                    # Save file
                    labels = dataset.validation_labels[i_val].astype(np.int32)
                    write_ply(val_name,
                              [points, preds, labels],
                              ['x', 'y', 'z', 'preds', 'class'])

                    i_val += 1

        return

    def multi_validation_error(self, model, dataset):
        """
        Validation method for multi object segmentation models
        """

        ##########
        # Initiate
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = []
            for p_l, o_l in zip(dataset.input_point_labels['validation'], dataset.input_labels['validation']):
                self.val_probs += [np.zeros((len(p_l), dataset.num_parts[o_l]))]

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        objects = []
        obj_inds = []
        mean_dt = np.zeros(2)
        last_display = time.time()
        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (model.logits,
                       model.labels,
                       model.inputs['super_labels'],
                       model.inputs['object_inds'],
                       model.inputs['in_batches'])
                prob, labels, object_labels, o_inds, batches = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    obj = object_labels[b[0]]
                    pred = prob[b][:, :model.config.num_classes[obj]]

                    # Stack all results
                    objects += [obj]
                    obj_inds += [o_inds[b_i]]
                    probs += [prob[b, :model.config.num_classes[obj]]]
                    targets += [labels[b]]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        ###################
        # Voting validation
        ###################

        for o_i, o_probs in zip(obj_inds, probs):
            self.val_probs[o_i] = val_smooth * self.val_probs[o_i] + (1 - val_smooth) * o_probs

        ############
        # Confusions
        ############

        # Confusion matrix for each object
        n_objs = [np.sum(np.array(objects) == l) for l in dataset.label_values]
        Confs = [np.zeros((n_obj, n_parts, n_parts), dtype=np.int32) for n_parts, n_obj in
                 zip(dataset.num_parts, n_objs)]
        obj_count = [0 for _ in n_objs]
        for obj, pred, truth in zip(objects, probs, targets):
            parts = [i for i in range(pred.shape[1])]
            Confs[obj][obj_count[obj], :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)
            obj_count[obj] += 1

        # Objects mIoU
        IoUs = [IoU_from_confusions(C) for C in Confs]


        # Compute votes confusion
        n_objs = [np.sum(np.array(dataset.input_labels['validation']) == l) for l in dataset.label_values]
        Confs = [np.zeros((n_obj, n_parts, n_parts), dtype=np.int32) for n_parts, n_obj in
                 zip(dataset.num_parts, n_objs)]
        obj_count = [0 for _ in n_objs]
        for obj, pred, truth in zip(dataset.input_labels['validation'],
                                    self.val_probs,
                                    dataset.input_point_labels['validation']):
            parts = [i for i in range(pred.shape[1])]
            Confs[obj][obj_count[obj], :, :] = confusion_matrix(truth, np.argmax(pred, axis=1), parts)
            obj_count[obj] += 1

        # Objects mIoU
        vote_IoUs = [IoU_from_confusions(C) for C in Confs]

        # Saving (optionnal)
        if model.config.saving:

            IoU_list = [IoUs, vote_IoUs]
            file_list = ['val_IoUs.txt', 'vote_IoUs.txt']

            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(model.saving_path, IoU_file)

                # Line to write:
                line = ''
                for obj_IoUs in IoUs_to_save:
                    for part_IoUs in obj_IoUs:
                        for IoU in part_IoUs:
                            line += '{:.3f} '.format(IoU)
                    line += '/ '
                line = line[:-2] + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in IoUs]))
        class_mIoUs = [np.mean(obj_IoUs) for obj_IoUs in IoUs]
        mcIoU = 100 * np.mean(class_mIoUs)
        print('Val  : mIoU = {:.1f}% / mcIoU = {:.1f}% '.format(mIoU, mcIoU))
        mIoU = 100 * np.mean(np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in vote_IoUs]))
        class_mIoUs = [np.mean(obj_IoUs) for obj_IoUs in vote_IoUs]
        mcIoU = 100 * np.mean(class_mIoUs)
        print('Vote : mIoU = {:.1f}% / mcIoU = {:.1f}% '.format(mIoU, mcIoU))

        return

    # Saving methods
    # ------------------------------------------------------------------------------------------------------------------

    def save_kernel_points(self, model, epoch):
        """
        Method saving kernel point disposition and current model weights for later visualization
        """

        if model.config.saving:

            # Create a directory to save kernels of this epoch
            kernels_dir = join(model.saving_path, 'kernel_points', 'epoch{:d}'.format(epoch))
            if not exists(kernels_dir):
                makedirs(kernels_dir)

            # Get points
            all_kernel_points_tf = [v for v in tf.global_variables() if 'kernel_points' in v.name
                                    and v.name.startswith('KernelPoint')]
            all_kernel_points = self.sess.run(all_kernel_points_tf)

            # Get Extents
            if False and 'gaussian' in model.config.convolution_mode:
                all_kernel_params_tf = [v for v in tf.global_variables() if 'kernel_extents' in v.name
                                        and v.name.startswith('KernelPoint')]
                all_kernel_params = self.sess.run(all_kernel_params_tf)
            else:
                all_kernel_params = [None for p in all_kernel_points]

            # Save in ply file
            for kernel_points, kernel_extents, v in zip(all_kernel_points, all_kernel_params, all_kernel_points_tf):

                # Name of saving file
                ply_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.ply'
                ply_file = join(kernels_dir, ply_name)

                # Data to save
                if kernel_points.ndim > 2:
                    kernel_points = kernel_points[:, 0, :]
                if False and 'gaussian' in model.config.convolution_mode:
                    data = [kernel_points, kernel_extents]
                    keys = ['x', 'y', 'z', 'sigma']
                else:
                    data = kernel_points
                    keys = ['x', 'y', 'z']

                # Save
                write_ply(ply_file, data, keys)

            # Get Weights
            all_kernel_weights_tf = [v for v in tf.global_variables() if 'weights' in v.name
                                    and v.name.startswith('KernelPointNetwork')]
            all_kernel_weights = self.sess.run(all_kernel_weights_tf)

            # Save in numpy file
            for kernel_weights, v in zip(all_kernel_weights, all_kernel_weights_tf):
                np_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.npy'
                np_file = join(kernels_dir, np_name)
                np.save(np_file, kernel_weights)

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def show_memory_usage(self, batch_to_feed):

            for l in range(self.config.num_layers):
                neighb_size = list(batch_to_feed[self.in_neighbors_f32[l]].shape)
                dist_size = neighb_size + [self.config.num_kernel_points, 3]
                dist_memory = np.prod(dist_size) * 4 * 1e-9
                in_feature_size = neighb_size + [self.config.first_features_dim * 2**l]
                in_feature_memory = np.prod(in_feature_size) * 4 * 1e-9
                out_feature_size = [neighb_size[0], self.config.num_kernel_points, self.config.first_features_dim * 2**(l+1)]
                out_feature_memory = np.prod(out_feature_size) * 4 * 1e-9

                print('Layer {:d} => {:.1f}GB {:.1f}GB {:.1f}GB'.format(l,
                                                                   dist_memory,
                                                                   in_feature_memory,
                                                                   out_feature_memory))
            print('************************************')

    def debug_nan(self, model, inputs, logits):
        """
        NaN happened, find where
        """

        print('\n\n------------------------ NaN DEBUG ------------------------\n')

        # First save everything to reproduce error
        file1 = join(model.config.saving_path, 'all_debug_inputs.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(inputs, f1)

        # First save all inputs
        file1 = join(model.config.saving_path, 'all_debug_logits.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(logits, f1)

        # Then print a list of the trainable variables and if they have nan
        print('List of variables :')
        print('*******************\n')
        all_vars = self.sess.run(tf.global_variables())
        for v, value in zip(tf.global_variables(), all_vars):
            nan_percentage = 100 * np.sum(np.isnan(value)) / np.prod(value.shape)
            print(v.name, ' => {:.1f}% of values are NaN'.format(nan_percentage))


        print('Inputs :')
        print('********')

        #Print inputs
        nl = model.config.num_layers
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
            nan_percentage = 100 * np.sum(np.isnan(pools)) / np.prod(pools.shape)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / np.prod(upsamples.shape)
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
        if model.config.dataset.startswith('ShapeNetPart_multi'):
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
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            max_n = np.max(pools)
            nums = np.sum(pools < max_n - 0.5, axis=-1)
            print('min pools =>', np.min(nums))

            max_n = np.max(upsamples)
            nums = np.sum(upsamples < max_n - 0.5, axis=-1)
            print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')
        time.sleep(0.5)



































