#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Classification model
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
import numpy as np

# Convolution functions
from models.network_blocks import assemble_CNN_blocks, classification_head, classification_loss


# ----------------------------------------------------------------------------------------------------------------------
#
#           Model Class
#       \*****************/
#


class KernelPointCNN:

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
            ind = 3 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['labels'] = flat_inputs[ind]
            ind += 1
            self.labels = self.inputs['labels']
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['object_inds'] = flat_inputs[ind]

            # Dropout placeholder
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        ########
        # Layers
        ########

        # Create layers
        with tf.variable_scope('KernelPointNetwork'):
            F = assemble_CNN_blocks(self.inputs,
                                    self.config,
                                    self.dropout_prob)

            self.logits = classification_head(F[-1],
                                              self.config,
                                              self.dropout_prob)

        ########
        # Losses
        ########

        with tf.variable_scope('loss'):

            # Classification loss
            self.output_loss = classification_loss(self.logits,
                                                   self.inputs)

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
                target_extent = np.float32(1.0 * conv_radius / np.power(self.config.num_kernel_points, 1 / 3))
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
                    conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** layer)

                    # Get the distance to closest input point
                    KP_min_d2 = tf.reduce_min(deformed_d2, axis=1)

                    # Normalize KP locations to be independant from layers
                    KP_min_d2 = KP_min_d2 / (conv_radius**2)

                    # Loss will be the square distance to closest input point.
                    # Mean => loss independent from the number of input points
                    offset_losses += [tf.reduce_mean(KP_min_d2)]

        elif self.config.offsets_loss != 'none':
            raise ValueError('Unknown offset loss')

        if len(offset_losses) > 0:
            self.offsets_loss = self.config.offsets_decay * tf.add_n(offset_losses)
        else:
            self.offsets_loss = tf.constant(0, dtype=tf.float32)

        return self.offsets_loss + self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)












"""
Hi,

As explained in my last mail, I implemented a deformable KPConv following the two articles DeformableConv and DeformableConvV2. This mail details the new desing.

First, I modified our KPConv design so that it would be adapted to deformable convolution differences. Here are the differences:
Until now, the KPConv had modes (closest, gaussian). I decided to separate the mode in two notions : the influence function (which can be gaussian or constant) and the aggregation mode (which can be summation or closest point). The previous Gfix mode thus corresponds to a Gaussian influence with summation aggregation and the previous Closest mode to a constant influence with closest aggregation

Following the billinear interpolation done in DeformableConv, I added a new influence function called "linear" and defined as h(x, x_k) = max(0, 1 - ||x - x_k|| / KP_extent). This influence decrease linearly with the distance (like the Gaussian) and saturate at zero when distance is larger than the parameter called "KP_extent".

The new parameter KP_extent controls the influence radius of each Kernel Point. It is used to define the influence functions
Extent of the linear influence like described above.
Sigma of the Gaussian: sigma = 0.3 * KP_Extent.
Range of the constant: influence is set to 1 if ||x - x_k|| < KP_extent, 0 otherwise
KP_extent is set to 1.0 * grid_dl, in the same way as the DeformableConv article, whose bilinear interpolation has a range of 1 pixel.
For efficiency, any neighbor "x" whose distance to all kernel points is greater than KP_extent is ignored. This does not change anything in the normal KPConv but is crucial for deformable KPConv (see below)
This new design is more clear than the "modes" we defined before, and is suited for the deformable convolutions. Now that we control the KP_extent with a parameter, we can set the positions of the kernel points according to it (I currently ensure that the kernel points are placed at 1.5 * KP_extent from the center of the kernel). Our old parameter called "density_parameter" (ratio between the input neighborhood radius and the subsampling grid size) now only controls the radius of the input sphere, while KP_extent controls the radius of our kernel effectively applied. I normal KPConv, the density parameter is useless, but it will be used for deformable convolutions.

Then, I defined a deformable KPConv operator:
The first step is to apply a normal KPConv that outputs a 3D offset for each kernel points. Optionally we can also output a set of weights (called modulations), that will modulated the impact of each offsetted KP. Like in the DeformableConv article, we scale the gradient of these features by 0.1. A slight difference with the original article, I rescale the offsets by a multiplication with KP_extent, so that the learned offsets are independant from the layer scale.

Now the second step is to apply a deformed KPConv. We simply compute the distances from input neighbors to the deformed kernel points instead of the original ones, and the rest of the convolution is almost identical (the only other difference is the modulation that is optionally applied when summing the kernel point features)

As we notice in the DeformableConv paper, one of the main strength of the deformable kernel is its ability to behave like an atrous convolution with a wider receptive field. This is where the "density_parameter" comes in handy. We cannot afford to compute the distances to every points of the input, so we still need to bound the possible range of a deformable KPConv, but we can set the density parameter to 4.0, meaning that the deformed kernel points will be able to "see" input points up to 4 times the subsampling grid size

applied KP weights to every neighbors at this location, but some of the weights would be zeros in case of closest mode, or close to zeros in case of gaussian mode. I decided to split 


"""





















