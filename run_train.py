import os
import sys
import math
import random
import argparse

import numpy as np
import tensorflow_addons as tfa
import tensorflow.compat.v1 as tf

from copy import deepcopy
from datetime import datetime
from typing import List
from time import time
from pathlib import Path
from configparser import ConfigParser

sys.path.append(str(Path.cwd().parent))

from utils_training import *
from network import Res2Net
from label_calcu import parse_pdb_6d
from arguments import get_args

########################################################
# read args and hyparams
########################################################
parser = argparse.ArgumentParser()

parser.add_argument('-npz', '--save_npz_pth', type=str, required=True, help='(input) path to npz files storing MSA and labels parsing from A3M & PDB files', default='../training_set/npz/')
parser.add_argument('-model', '--out_pth', type=str, default='models/tr_res2net_v2', help='(output) path to store ckpt files and training log')
parser.add_argument('-gpu', '--gpu_id', type=str, default=0, help='use which GPU')
parser.add_argument('--early_stopping', type=bool, default=True, help='whether to stop early if val loss cannot drop')

args = parser.parse_args()

output_pth = args.out_pth
gpu_id = args.gpu_id
npz_pth = args.save_npz_pth
init_epoch = 0
early_stopping = args.early_stopping

model_name = 'res2net_v2'

config = ConfigParser()
config.read('config.ini')
hyparams = dict((k, float(v)) for (k, v) in config['hyparams'].items())
n_bins = dict((k, int(v)) for (k, v) in config['n_bins'].items())

if early_stopping:
    loss_cutoff = hyparams['early_stopping_cutoff']
    max_epoch_without_improve = hyparams['max_epoch_without_improve']
else:
    loss_cutoff = -np.inf
    max_epoch_without_improve = 100

log_pth = f'{output_pth}/training.log'
long_list_file = f'{npz_pth}/long_list_all'

########################################################
# env and tf config
########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel("ERROR")
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6, allow_growth=True)
)

opt = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)  # show allocations of tensors' memory when oom happened

########################################################
# network
########################################################
wmin = 0.8
ns = 21

activation = tf.nn.elu
conv1d = tf.compat.v1.layers.conv1d
conv2d = tf.compat.v1.layers.conv2d
# normalize = tf.keras.layers.instance_norm  # tf1
normalize = tfa.layers.InstanceNormalization()  # tf1


def main():
    with tf.Graph().as_default():
        reg = tf.keras.regularizers.l2(0.5 * (hyparams['reg_rate']))
        with tf.compat.v1.name_scope('input'):
            ncol = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name='ncol')
            nrow = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name='nrow')
            n_sub = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name='n_sub')
            msa = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, None), name='msa')
            learning_rate_decay = tf.compat.v1.placeholder(shape=(), dtype=tf.float32, name='learning_rate_decay')

            theta = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None, None))
            phi = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None, None))
            distance = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None, None))
            omega = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, None, None))

        with tf.compat.v1.name_scope('preprocess'):
            y_distance = tf.one_hot(distance, n_bins['distance'])
            y_theta = tf.one_hot(theta, n_bins['theta'])
            y_phi = tf.one_hot(phi, n_bins['phi'])
            y_omega = tf.one_hot(omega, n_bins['omega'])

            def collect_features(msa_sample):
                msa1hot = tf.one_hot(msa_sample, ns, dtype=tf.float32)
                w = reweight(msa1hot, wmin)

                # 1D features
                f1d_seq = msa1hot[0, :, :20]
                f1d_pssm = msa2pssm(msa1hot, w)

                f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
                f1d = tf.expand_dims(f1d, axis=0)
                f1d = tf.reshape(f1d, [1, ncol, 42])

                # 2D features
                f2d_dca = tf.cond(pred=nrow > 1, true_fn=lambda: fast_dca(msa1hot, w), false_fn=lambda: tf.zeros([ncol, ncol, 442], tf.float32))
                f2d_dca = tf.expand_dims(f2d_dca, axis=0)

                f2d = tf.concat([tf.tile(f1d[:, :, None, :], [1, 1, ncol, 1]),
                                 tf.tile(f1d[:, None, :, :], [1, ncol, 1, 1]),
                                 f2d_dca], axis=-1)
                return tf.reshape(f2d, [ncol, ncol, 442 + 2 * 42])

            f2d = tf.map_fn(lambda x: collect_features(x), msa, dtype=tf.float32)

        with tf.compat.v1.name_scope('network'):
            # res2net = Res2Net(reg_rate=reg).forward
            model = Res2Net(reg_rate=reg)
            res2net = model.forward
            # model.summary()
            print('f2d:',f2d.shape)
            output_tensor = res2net(f2d, layer=int(hyparams['n_layers']))
            output_tensor = activation(normalize(output_tensor))

            # print('output_tensor:',output_tensor.shape)

            # symmetrize
            symmetrized = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3]) + output_tensor

            # output layer
            softmax_theta = tf.nn.softmax(conv2d(output_tensor, n_bins['theta'], 1, kernel_regularizer=reg))
            softmax_phi = tf.nn.softmax(conv2d(output_tensor, n_bins['phi'], 1, kernel_regularizer=reg))
            softmax_distance = tf.nn.softmax(conv2d(symmetrized, n_bins['distance'], 1, kernel_regularizer=reg))
            softmax_omega = tf.nn.softmax(conv2d(symmetrized, n_bins['omega'], 1, kernel_regularizer=reg))

            tf.identity(softmax_phi, 'pred_phi')
            tf.identity(softmax_theta, 'pred_theta')
            tf.identity(softmax_distance, 'pred_distance')
            tf.identity(softmax_omega, 'pred_omega')

            # print('softmax_phi:',softmax_phi.shape)
            # print('y_theta:',y_phi.shape)
            # calculate loss function
            loss_branches = [
                -tf.reduce_sum(input_tensor=tf.math.log(softmax_theta + 1e-8) * y_theta, axis=-1),
                -tf.reduce_sum(input_tensor=tf.math.log(softmax_phi + 1e-8) * y_phi, axis=-1),
                -tf.reduce_sum(input_tensor=tf.math.log(softmax_distance + 1e-8) * y_distance, axis=-1),
                -tf.reduce_sum(input_tensor=tf.math.log(softmax_omega + 1e-8) * y_omega, axis=-1),
            ]

            all_loss = [tf.reduce_mean(input_tensor=l) for l in loss_branches]
            distance_loss = all_loss[2]
            total_loss = tf.reduce_mean(input_tensor=all_loss)
            regularization_loss = tf.reduce_sum(input_tensor=tf.compat.v1.losses.get_regularization_losses())
            total_loss += regularization_loss

            # optimization
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hyparams['init_learning_rate'] * learning_rate_decay)
            optimizer = optimizer.minimize(total_loss)

        def validation(sess):
            """
            do validation after each epoch
            :param sess: tf.Session storing the training params
            :return: val_loss, val_precision, val_dist_loss
            """
            n_ = len(val_set)
            losses_, accuracies_, dist_losses_ = [], [], []
            n_OOM_, min_OOM_shape_ = 0, np.inf

            for i, pid in enumerate(val_set):
                npz_file = f'{npz_pth}/{pid}.npz'
                npz = np.load(npz_file)
                a3m = npz['msa'][:20000, :]
                theta_ = npz['theta_asym'][None, ...]
                phi_ = npz['phi_asym'][None, ...]
                dist_ = npz['dist'][None, ...]
                omega_ = npz['omega'][None, ...]

                a3m = a3m[None, ...]

                dbin, obin, tbin, pbin = binning(dist_, omega_, theta_, phi_)

                try:
                    pred_dist, loss_, dist_loss_ = sess.run([softmax_distance, total_loss, distance_loss], options=opt,
                                                            feed_dict={msa: a3m,
                                                                       ncol: a3m.shape[-1], nrow: a3m.shape[-2], n_sub: 1,
                                                                       theta: tbin, phi: pbin,
                                                                       distance: dbin, omega: obin,
                                                                       learning_rate_decay: 0})
                    losses_.append(loss_)
                    dist_losses_.append(dist_loss_)
                    accuracy_ = cont_acc(pred_dist, dist_[0], dist_type='v')
                    accuracies_.append(accuracy_)

                except tf.errors.ResourceExhaustedError:
                    # if out of GPU memory, record the length
                    n_OOM_ += 1
                    if a3m.shape[1] < min_OOM_shape_: min_OOM_shape_ = a3m.shape[1]
                    continue
                print('\r',
                      f'{i + 1}/{n_}, val_loss: {loss_:.3f},val_accuracy:{accuracy_:.4f}, n_OOM:{n_OOM_}, min_OOM_len:{min_OOM_shape_}',
                      end="")
            return np.mean(losses_), np.mean(accuracies_), np.mean(dist_losses_)

        os.makedirs(output_pth, exist_ok=True)
        try:
            long_list = [p.strip() for p in open(long_list_file).readlines()]
            n_da = int(hyparams['data_augmentation'])
        except FileNotFoundError:
            long_list = []
            n_da = 0
            print('warning: long_list_all missed!')

        # split samples for training, validation
        files_list = [f.split('.')[0] for f in os.listdir(npz_pth) if f.endswith('.npz')]
        if init_epoch == 0:
            val_set = random.sample(files_list, int(len(files_list) * hyparams['val_prop']))
            with open(f'{output_pth}/val_list_{str(datetime.now().date())}', 'w') as f:
                f.write('\n'.join(val_set))

        else:
            val_list = [f for f in os.listdir(output_pth) if f.startswith('val')][-1]
            val_set = [f.strip() for f in open(f'{output_pth}/{val_list}')]
        train_set = set(files_list) - set(val_set)

        # for data augmentation
        long_list = [f for f in long_list if f.replace('_long', '') not in val_set]
        print(long_list.__len__())

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open(log_pth, 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(f'\n---------------------------------------------------learning_rate:{hyparams["init_learning_rate"]}---------------------------------------------------')
            sys.stdout = original_stdout  # Reset the standard output to its original value

        #查看
        tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)
        print("is_gpu: ", tf.test.is_gpu_available())
        
        ########################################################
        # training
        ########################################################
        with tf.compat.v1.Session(config=config) as session:
            # initializer params
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.local_variables_initializer())

            saver = tf.compat.v1.train.Saver(max_to_keep=50)
            if init_epoch > 0:
                saver.restore(session, f'{output_pth}/checkpoint_dir/epoch_{init_epoch}')
            epoch_no_improve, min_loss = 0, np.inf
            print('--------------------------------train start-------------------------------------', end='\n')
            for epoch in range(init_epoch, int(hyparams['epochs'])):
                st = time()

                # data augmentation
                train_sample = random.sample(long_list, n_da) + list(train_set)

                random.shuffle(train_sample)
                n_samples = len(train_sample)
                n_OOM = 0
                min_OOM_shape = np.inf
                losses, accuracies = [], []
                dist_losses = []
                for i, pid in enumerate(train_sample):
                    # continue
                    with tf.device('/CPU:0'):
                        npz_file = f'{npz_pth}/{pid}.npz'
                        a3m, theta_, phi_, dist_, omega_ = read_single_file_train(npz_file, cutoff=int(hyparams['cutoff']))

                        dist_, omega_, theta_, phi_ = binning(dist_, omega_, theta_, phi_)
                        
                    # print('phi_:',phi_.shape)

                    try:
                        # Forward an backward propagation
                        loss, dist_loss, _ = session.run(
                            [total_loss, distance_loss, optimizer],
                            options=opt,
                            feed_dict={
                                msa: a3m,
                                ncol: a3m.shape[-1],
                                nrow: a3m.shape[-2],
                                n_sub: a3m.shape[0],
                                theta: theta_,
                                phi: phi_,
                                distance: dist_,
                                omega: omega_,
                                learning_rate_decay: lr_schedule(epoch)
                            })
                        losses.append(loss)
                        dist_losses.append(dist_loss)

                    except tf.errors.ResourceExhaustedError:
                        # if out of GPU memory, record the length
                        n_OOM += 1
                        if a3m.shape[-1] * a3m.shape[0] < min_OOM_shape:
                            min_OOM_shape = a3m.shape[-1] * a3m.shape[0]
                            continue
                    e = time()
                    if np.isnan(loss):
                        raise ValueError(f'The {i}th sample, loss is nan!')
                    print(
                        '\r',
                        f'epoch:{epoch}, {i + 1}/{n_samples}, loss:{loss:.3f},time:{np.round(e - st, 2)}s, n_OOM:{n_OOM},min_OOM_len:{min_OOM_shape}        ',
                        end="")
                # save ckpt for each epoch
                saver.save(session, f'{output_pth}/cp_dir/epoch_{epoch}')

                ########################################################
                # validation
                ########################################################

                val_loss, val_accuracy, val_dist_loss = validation(session)

                e = time()
                print("\r",
                      f"""---------------------------------------------------------------------------
epoch {epoch} ends, time:{np.round(e - st, 2)}s, n_OOM:{n_OOM}, min_len_for_OOM:{min_OOM_shape}
train: train_loss:{np.mean(losses):.3f}(dist:{np.mean(dist_losses):.3f})                                    
validation: val_loss:{val_loss:.3f}(dist_loss:{val_dist_loss}) long-range top L contact precision:{val_accuracy * 100:.2f}
""")
                with open(log_pth, 'a') as f:
                    sys.stdout = f  # Change the standard output to the file we created.
                    print(
                        f"epoch {epoch}|train_loss:{np.mean(losses):.3f} val_loss:{val_loss:.3f} val long-range top L contact precision:{val_accuracy * 100:.2f}")
                    sys.stdout = original_stdout  # Reset the standard output to its original value

                # check ealry stopping
                if val_loss > min_loss - loss_cutoff:
                    epoch_no_improve += 1
                    if val_loss < min_loss:
                        min_loss = val_loss
                        best_epoch = epoch
                else:
                    epoch_no_improve = 0
                    min_loss = val_loss
                    best_epoch = epoch

                if epoch_no_improve == max_epoch_without_improve:
                    print(f'val_loss cannot drop further, early stopping happens! The best epoch is {best_epoch}')
                    break
            # save best epoch
            saver.save(session, output_pth + '/' + model_name)
            saver.restore(session, f'{output_pth}/checkpoint_dir/epoch_{best_epoch}')
            print('--------------------------------train end---------------------------------------')


if __name__ == '__main__':
    main()
