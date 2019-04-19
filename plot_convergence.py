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
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd
from sklearn.metrics import confusion_matrix

# My libs
from utils.config import Config
from utils.metrics import IoU_from_confusions, smooth_metrics
from utils.ply import read_ply

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset
from datasets.S3DIS import S3DISDataset
from datasets.Scannet import ScannetDataset
from datasets.Semantic3D import Semantic3DDataset
from datasets.NPM3D import NPM3DDataset

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def running_mean(signal, n, axis=0):

    signal = np.array(signal)
    if signal.ndim == 1:
        signal_sum = np.convolve(signal, np.ones((2*n+1,)), mode='same')
        signal_num = np.convolve(signal*0+1, np.ones((2*n+1,)), mode='same')
        return signal_sum/signal_num

    elif signal.ndim == 2:
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_multi_metrics(all_IoUs, smooth_n):

    # Get mean IoU for consecutive epochs to directly get a mean
    all_mIoUs = [np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs]) for epoch_IoUs in all_IoUs]
    smoothed_mIoUs = []
    for epoch in range(len(all_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_mIoUs))
        smoothed_mIoUs += [np.mean(np.hstack(all_mIoUs[i0:i1]))]

    # Get mean for each class
    all_objs_mIoUs = [[np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs] for epoch_IoUs in all_IoUs]
    smoothed_obj_mIoUs = []
    for epoch in range(len(all_objs_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_objs_mIoUs))

        epoch_obj_mIoUs = []
        for obj in range(len(all_objs_mIoUs[0])):
            epoch_obj_mIoUs += [np.mean(np.hstack([objs_mIoUs[obj] for objs_mIoUs in all_objs_mIoUs[i0:i1]]))]

        smoothed_obj_mIoUs += [epoch_obj_mIoUs]

    return np.array(smoothed_mIoUs), np.array(smoothed_obj_mIoUs)


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    steps = []
    L_out = []
    L_reg = []
    L_p = []
    acc = []
    t = []
    memory = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            steps += [int(line_info[0])]
            L_out += [float(line_info[1])]
            L_reg += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
            memory += [float(line_info[6])]
        else:
            break

    return steps, L_out, L_reg, L_p, acc, t, memory


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        obj_IoUs = [[float(IoU) for IoU in s.split()] for s in line.split('/')]
        obj_IoUs = [np.reshape(IoUs, [-1, n_parts[obj]]) for obj, IoUs in enumerate(obj_IoUs)]
        all_IoUs += [obj_IoUs]
    return all_IoUs


def compare_trainings(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_epochs = 1

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []

    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(path)

        # Compute number of steps per epoch
        if config.epoch_steps is None:
            if config.dataset == 'ModelNet40':
                steps_per_epoch = np.ceil(9843 / int(config.batch_num))
            else:
                raise ValueError('Unsupported dataset')
        else:
            steps_per_epoch = config.epoch_steps

        smooth_n = int(steps_per_epoch * smooth_epochs)

        # Load results
        steps, L_out, L_reg, L_p, acc, t, memory = load_training_results(path)
        all_epochs += [np.array(steps) / steps_per_epoch]
        all_loss += [running_mean(L_out, smooth_n)]
        all_times += [t]

        # Learning rate
        lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
        lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
        max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
        lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
        lr_decays[0] = float(config.learning_rate)
        lr_decays[lr_decay_e] = lr_decay_v
        lr = np.cumprod(lr_decays)
        all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

    # Plots learning rate
    # *******************

    if False:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    # Figure
    fig = plt.figure('loss')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    # Display legends and title
    plt.legend(loc=1)
    plt.title('Losses compare')

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    fig = plt.figure('time')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('time')
    # plt.yscale('log')

    # Display legends and title
    plt.legend(loc=0)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_multisegment(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 10

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_instances_mIoUs = []
    all_objs_mIoUs = []
    all_objs_IoUs = []
    all_parts = []

    obj_list = ['Air', 'Bag', 'Cap', 'Car', 'Cha', 'Ear', 'Gui', 'Kni',
                'Lam', 'Lap', 'Mot', 'Mug', 'Pis', 'Roc', 'Ska', 'Tab']
    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')
    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(path)

        # Get the number of classes
        n_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        part = config.dataset.split('_')[-1]

        # Get validation confusions
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_multi_IoU(file, n_parts)

        file = join(path, 'vote_IoUs.txt')
        vote_IoUs = load_multi_IoU(file, n_parts)

        #print(len(val_IoUs[0]))
        #print(val_IoUs[0][0].shape)

        # Get mean IoU
        #instances_mIoUs, objs_mIoUs = IoU_multi_metrics(val_IoUs, smooth_n)

        # Get mean IoU
        instances_mIoUs, objs_mIoUs = IoU_multi_metrics(vote_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_instances_mIoUs += [instances_mIoUs]
        all_objs_IoUs += [objs_mIoUs]
        all_objs_mIoUs += [np.mean(objs_mIoUs, axis=1)]

        if part == 'multi':
            s = '{:4.1f} | {:4.1f} | '.format(100 * np.mean(objs_mIoUs[-1]), 100 * instances_mIoUs[-1])
            for obj_mIoU in objs_mIoUs[-1]:
                s += '{:4.1f} '.format(100 * obj_mIoU)
            print(s)
        else:
            s = ' --  |  --  | '
            for obj_name in obj_list:
                if part.startswith(obj_name):
                    s += '{:4.1f} '.format(100 * instances_mIoUs[-1])
                else:
                    s += ' --  '.format(100 * instances_mIoUs[-1])
            print(s)
        all_parts += [part]

    # Plots
    # *****

    if 'multi' in all_parts:

        # Figure
        fig = plt.figure('Instances mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_instances_mIoUs[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel('IoU')

        # Set limits for y axis
        #plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

        # Figure
        fig = plt.figure('mean of categories mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel('IoU')

        # Set limits for y axis
        #plt.ylim(0.8, 1)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    for obj_i, obj_name in enumerate(obj_list):
        if np.any([part.startswith(obj_name) for part in all_parts]):
            # Figure
            fig = plt.figure(obj_name + ' mIoU')
            for i, label in enumerate(list_of_labels):
                if all_parts[i] == 'multi':
                    plt.plot(all_pred_epochs[i], all_objs_IoUs[i][:, obj_i], linewidth=1, label=label)
                elif all_parts[i].startswith(obj_name):
                    plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


def compare_convergences_segment(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_mIoUs = []
    all_class_IoUs = []
    all_snap_epochs = []
    all_snap_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, config.num_classes)

        # Get mean IoU
        class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_mIoUs += [mIoUs]
        all_class_IoUs += [class_IoUs]

        s = '{:^10.1f}|'.format(100*mIoUs[-1])
        for IoU in class_IoUs[-1]:
            s += '{:^10.1f}'.format(100*IoU)
        print(s)

        # Get optional full validation on clouds
        snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
        all_snap_epochs += [snap_epochs]
        all_snap_IoUs += [snap_IoUs]

    print(10*'-' + '|' + 10*config.num_classes*'-')
    for snap_IoUs in all_snap_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^10.1f}'.format(100*IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


def compare_convergences_classif(dataset, list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 2

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_OA = []
    all_train_OA = []
    all_vote_OA = []
    all_vote_confs = []


    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(list_of_paths[0])

        # Get the number of classes
        n_class = config.num_classes

        # Get validation confusions
        file = join(path, 'val_confs.txt')
        val_C1 = load_confusions(file, n_class)
        val_PRE, val_REC, val_F1, val_IoU, val_ACC = smooth_metrics(val_C1, smooth_n=smooth_n)

        # Get vote confusions
        file = join(path, 'vote_confs.txt')
        if exists(file):
            vote_C2 = load_confusions(file, n_class)
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = smooth_metrics(vote_C2, smooth_n=2)
        else:
            vote_C2 = val_C1
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = (val_PRE, val_REC, val_F1, val_IoU, val_ACC)

        # Get training confusions balanced
        file = join(path, 'training_confs.txt')
        train_C = load_confusions(file, n_class)
        train_PRE, train_REC, train_F1, train_IoU, train_ACC = smooth_metrics(train_C, smooth_n=smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_ACC))])]
        all_val_OA += [val_ACC]
        all_vote_OA += [vote_ACC]
        all_train_OA += [train_ACC]
        all_vote_confs += [vote_C2]
        #all_mean_IoU_scores += [running_mean(np.mean(val_IoU[:, 1:], axis=1), smooth_n)]


    # Best scores
    # ***********

    for i, label in enumerate(list_of_labels):

        print('\n' + label + '\n' + '*' * len(label) + '\n')

        best_epoch = np.argmax(all_vote_OA[i])
        print('Best Accuracy : {:.1f} % (epoch {:d})'.format(100 * all_vote_OA[i][best_epoch], best_epoch))

        confs = all_vote_confs[i]
        TP_plus_FN = np.sum(confs, axis=-1, keepdims=True)
        class_avg_confs = confs.astype(np.float32) / TP_plus_FN.astype(np.float32)
        diags = np.diagonal(class_avg_confs, axis1=-2, axis2=-1)
        class_avg_ACC = np.sum(diags, axis=-1) / np.sum(class_avg_confs, axis=(-1, -2))

        print('Corresponding mAcc : {:.1f} %'.format(100 * class_avg_ACC[best_epoch]))

    # Plots
    # *****

    # Figure
    fig = plt.figure('Validation')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_val_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Validation Accuracy')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Figure
    fig = plt.figure('Vote Validation')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_vote_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Validation Accuracy')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Figure
    fig = plt.figure('Training')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_train_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Overall Accuracy')

    # Set limits for y axis
    #plt.ylim(0.8, 1)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))


    #for i, label in enumerate(list_of_labels):
    #    print(label, np.max(all_train_OA[i]), np.max(all_val_OA[i]))


    # Show all
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2018-12-12_17-42-20'
    end = 'Log_2018-12-12_17-42-21'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['Log1_rigid',
                  'Log2_deform']
    logs_names = np.array(logs_names[:len(logs)])

    ################################################################
    # The right plotting function is called depending on the dataset
    ################################################################

    # Check that all logs are of the same dataset. Different object can be compared
    plot_dataset = None
    for log in logs:
        config = Config()
        config.load(log)
        if plot_dataset:
            if plot_dataset in config.dataset:
                continue
            else:
                raise ValueError('All logs must share the same dataset to be compared')
        else:
            plot_dataset = config.dataset[:5]

    # Plot the training loss and accuracy
    compare_trainings(logs, logs_names)

    # Plot the validation
    if plot_dataset.startswith('Shape'):
        compare_convergences_multisegment(logs, logs_names)
    elif plot_dataset.startswith('S3DIS'):
        dataset = S3DISDataset()
        compare_convergences_segment(dataset, logs, logs_names)
    elif plot_dataset.startswith('Model'):
        dataset = ModelNet40Dataset()
        compare_convergences_classif(dataset, logs, logs_names)
    elif plot_dataset.startswith('Scann'):
        dataset = ScannetDataset()
        compare_convergences_segment(dataset, logs, logs_names)
    elif plot_dataset.startswith('Seman'):
        dataset = Semantic3DDataset()
        compare_convergences_segment(dataset, logs, logs_names)
    elif plot_dataset.startswith('NPM3D'):
        dataset = NPM3DDataset()
        compare_convergences_segment(dataset, logs, logs_names)
    else:
        raise ValueError('Unsupported dataset : ' + plot_dataset)


