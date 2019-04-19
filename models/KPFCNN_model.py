#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Segmentation model
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
from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import sys

# Convolution functions
from models.network_blocks import assemble_FCNN_blocks, segmentation_head, multi_segmentation_head
from models.network_blocks import segmentation_loss, multi_segmentation_loss


# ----------------------------------------------------------------------------------------------------------------------
#
#           Model Class
#       \*****************/
#


class KernelPointFCNN:

    def __init__(self, flat_inputs, config):
        """
        Initiate the model
        :param flat_inputs: List of input tensors (flatten)
        :param config: configuration class
        """

        # Model parameters
        self.config = config

        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path == None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            if not exists(self.saving_path):
                makedirs(self.saving_path)

        ########
        # Inputs
        ########

        # Sort flatten inputs in a dictionary
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.labels = self.inputs['point_labels']

            if config.network_model in ['multi_segmentation', 'multi_cloud_segmentation']:
                self.inputs['super_labels'] = flat_inputs[ind]
                ind += 1

            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]

            if config.network_model in ["cloud_segmentation", 'multi_cloud_segmentation']:
                ind += 1
                self.inputs['point_inds'] = flat_inputs[ind]
                ind += 1
                self.inputs['cloud_inds'] = flat_inputs[ind]

            elif config.network_model in ['multi_segmentation', 'segmentation']:
                ind += 1
                self.inputs['object_inds'] = flat_inputs[ind]

            # Dropout placeholder
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        ########
        # Layers
        ########

        # Create layers
        with tf.variable_scope('KernelPointNetwork'):
            output_features = assemble_FCNN_blocks(self.inputs,
                                                   self.config,
                                                   self.dropout_prob)

            if config.network_model in ["multi_segmentation", 'multi_cloud_segmentation']:
                self.logits = multi_segmentation_head(output_features,
                                                      self.inputs['super_labels'],
                                                      self.config,
                                                      self.dropout_prob)
            else:
                self.logits = segmentation_head(output_features,
                                                self.config,
                                                self.dropout_prob)

        ########
        # Losses
        ########

        with tf.variable_scope('loss'):


            if config.network_model in ["multi_segmentation", 'multi_cloud_segmentation']:
                self.output_loss = multi_segmentation_loss(self.logits,
                                                           self.inputs,
                                                           batch_average=self.config.batch_averaged_loss)

            elif len(self.config.ignored_label_inds) > 0:

                # Boolean mask of points that should be ignored
                ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
                for ign_label in self.config.ignored_label_inds:
                    ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

                # Collect logits and labels that are not ignored
                inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
                new_logits = tf.gather(self.logits, inds, axis=0)
                new_dict = {'point_labels': tf.gather(self.labels, inds, axis=0)}

                # Reduce label values in the range of logit shape
                reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
                inserted_value = tf.zeros((1,), dtype=tf.int32)
                for ign_label in self.config.ignored_label_inds:
                    reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
                new_dict['point_labels'] = tf.gather(reducing_list, new_dict['point_labels'])

                # Add batch weigths to dict if needed
                if self.config.batch_averaged_loss:
                    new_dict['batch_weights'] = self.inputs['batch_weights']

                # Output loss
                self.output_loss = segmentation_loss(new_logits,
                                                     new_dict,
                                                     batch_average=self.config.batch_averaged_loss)

            else:
                self.output_loss = segmentation_loss(self.logits,
                                                     self.inputs,
                                                     batch_average=self.config.batch_averaged_loss)

            # Add regularization
            self.loss = self.regularization_losses() + self.output_loss

        return

    def regularization_losses(self):

        #####################
        # Regularization loss
        #####################

        # Get L2 norm of all weights
        regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
        self.regularization_loss = self.config.weights_decay * tf.add_n(regularization_losses)

        ##############################
        # Gaussian regularization loss
        ##############################

        gaussian_losses = []
        for v in tf.global_variables():
            if 'kernel_extents' in v.name:

                # Layer index
                layer = int(v.name.split('/')[1].split('_')[-1])

                # Radius of convolution for this layer
                conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** (layer - 1))

                # Target extent
                target_extent = conv_radius / 1.5
                gaussian_losses += [tf.nn.l2_loss(v - target_extent)]

        if len(gaussian_losses) > 0:
            self.gaussian_loss = self.config.gaussian_decay * tf.add_n(gaussian_losses)
        else:
            self.gaussian_loss = tf.constant(0, dtype=tf.float32)

        #############################
        # Offsets regularization loss
        #############################

        offset_losses = []

        if self.config.offsets_loss == 'permissive':

            for op in tf.get_default_graph().get_operations():
                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_positions = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_positions/conv_radius

                    # Loss will be zeros inside radius and linear outside radius
                    # Mean => loss independent from the number of input points
                    radius_outside = tf.maximum(0.0, tf.norm(KP_locs, axis=2) - 1.0)
                    offset_losses += [tf.reduce_mean(radius_outside)]


        elif self.config.offsets_loss == 'fitting':

            for op in tf.get_default_graph().get_operations():

                if op.name.endswith('deformed_d2'):

                    # Get deformed distances
                    deformed_d2 = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Get the distance to closest input point
                    KP_min_d2 = tf.reduce_min(deformed_d2, axis=1)

                    # Normalize KP locations to be independant from layers
                    KP_min_d2 = KP_min_d2 / (KP_extent**2)

                    # Loss will be the square distance to closest input point.
                    # Mean => loss independent from the number of input points
                    offset_losses += [tf.reduce_mean(KP_min_d2)]

                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_KP = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_KP/KP_extent

                    # Point should not be close to each other
                    for i in range(self.config.num_kernel_points):
                        other_KP = tf.stop_gradient(tf.concat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], axis=1))
                        distances = tf.sqrt(tf.reduce_sum(tf.square(other_KP - KP_locs[:, i:i+1, :]), axis=2))
                        repulsive_losses = tf.reduce_sum(tf.square(tf.maximum(0.0, 1.5 - distances)), axis=1)
                        offset_losses += [tf.reduce_mean(repulsive_losses)]

        elif self.config.offsets_loss != 'none':
            raise ValueError('Unknown offset loss')

        if len(offset_losses) > 0:
            self.offsets_loss = self.config.offsets_decay * tf.add_n(offset_losses)
        else:
            self.offsets_loss = tf.constant(0, dtype=tf.float32)

        return self.offsets_loss + self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)


























