import random, os
import string, re

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# read A3M and convert letters into
# integers in the 0..20 range
# from tensorflow.python.layers.convolutional import Conv2D, conv2d


def parse_a3m(filename, limit=20000):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase + '*'))

    seq_len = len(open(filename, "r").readlines()[1]) - 1
    # read file line by line
    count = 0
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            line = line.rstrip().translate(table)
            if len(line) != seq_len:
                continue
            seqs.append(line.rstrip().translate(table))
            count += 1
            if count >= limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(input_tensor=w)
    f_i = tf.reduce_sum(input_tensor=w[:, None, None] * msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum(input_tensor=-f_i * tf.math.log(f_i), axis=1)
    return tf.concat([f_i, h_i[:, None]], axis=1)


# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.compat.v1.name_scope('reweight'):
        id_min = tf.cast(tf.shape(input=msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / tf.maximum(tf.reduce_sum(input_tensor=tf.cast(id_mask, dtype=tf.float32), axis=-1), .1)
    return w


# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty=4.5):
    nr = tf.shape(input=msa1hot)[0]
    nc = tf.shape(input=msa1hot)[1]
    ns = tf.shape(input=msa1hot)[2]

    with tf.compat.v1.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(input_tensor=weights) - tf.sqrt(tf.reduce_mean(input_tensor=weights))
        mean = tf.reduce_sum(input_tensor=x * weights[:, None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:, None])
        cov = tf.matmul(tf.transpose(a=x), x) / num_points

    with tf.compat.v1.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(input_tensor=weights))
        inv_cov = tf.linalg.inv(cov_reg)

        x1 = tf.reshape(inv_cov, (nc, ns, nc, ns))
        x2 = tf.transpose(a=x1, perm=[0, 2, 1, 3])
        features = tf.reshape(x2, (nc, nc, ns * ns))

        x3 = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(x1[:, :-1, :, :-1]), axis=(1, 3))) * (1 - tf.eye(nc))
        apc = tf.reduce_sum(input_tensor=x3, axis=0, keepdims=True) * tf.reduce_sum(input_tensor=x3, axis=1, keepdims=True) / tf.reduce_sum(input_tensor=x3)
        contacts = (x3 - apc) * (1 - tf.eye(nc))

    return tf.concat([features, contacts[:, :, None]], axis=2)


def cont_acc(pred, dist, gap=24, frac=1, dist_type='v'):
    nc = dist.shape[0]
    if len(pred.shape) == 4:
        w = np.sum(pred[0, :, :, 1:13], axis=-1)
    elif len(pred.shape) == 3:
        w = np.sum(pred[:, :, 1:13], axis=-1)
    else:
        raise ValueError('pred shape error!')
    idx = np.array([[i, j, w[i, j]] for i in range(nc) for j in range(i + gap, nc)])
    top = idx[np.argsort(idx[:, 2])][-int(nc / frac):, :2].astype(int)
    if dist_type == 'bin':
        ngood = np.sum([(dist[r[0], r[1]] < 13) & (dist[r[0], r[1]] >= 1) for r in top])
    else:
        ngood = np.sum([dist[r[0], r[1]] < 8.0 for r in top])
    return ngood / int(nc / frac)


def subsample(msa, limit=np.inf):
    nr, nc = msa.shape
    if nr < 10: return msa
    n = min(limit, int(10 ** np.random.uniform(np.log10(nr)) - 10))
    if n <= 0: return np.array([msa[0]])
    indices = sorted(random.sample(range(1, nr), n) + [0])
    return msa[indices]


def read_single_file_train(*input_files, cutoff=260, frac=130, msa_subsample=True, msa_limit=20000):
    """
    read and parse training sample
    :param a3m_file: input A3M file
    :param pdb_file: input PDB file
    :param cutoff: if length > cutoff, random sample one or two sub-sequences as training sample.
    :param frac: the length of each sub-sequence when doing binary sub-sampling.
    :param msa_subsample: whether to subsample MSA
    :param msa_limit: max number of homologous sequences to use
    :return: numpy arrays storing MSA & labels.
    """
    if len(input_files) == 2:
        npz, temp_npz = input_files
    else:
        npz, temp_npz = input_files[0], None

    label = np.load(npz.replace('_long', ''))
    a3m = label['msa']
    rrcs = label['rrcs']
    theta_ = label['theta_asym']
    phi_ = label['phi_asym']
    dist_ = label['dist']
    omega_ = label['omega']
    l = a3m.shape[-1]

    if temp_npz is not None:
        if os.path.isfile(temp_npz):
            temp_feats = np.load(temp_npz.replace('_long', ''))
            theta_temp_ = temp_feats['theta_asym']
            phi_temp_ = temp_feats['phi_asym']
            dist_temp_ = temp_feats['dist']
            omega_temp_ = temp_feats['omega']
        else:
            dist_temp_ = np.zeros((1, l, l))
            theta_temp_ = np.zeros((1, l, l))
            omega_temp_ = np.zeros((1, l, l))
            phi_temp_ = np.zeros((1, l, l))

    if msa_subsample:
        a3m = subsample(a3m, limit=msa_limit)
    else:
        a3m = a3m[:msa_limit]

    if l > cutoff:
        if 'long' in input_files[0]:
            frac = int(min(cutoff / 2, frac))
            left = max(0, int(10 ** np.random.uniform(np.log10(l / 2 - frac)) - 10))
            right = l - max(int(10 ** np.random.uniform(np.log10(l / 2 - frac)) - 10), 0)
            ind = list(range(left, left + frac)) + list(range(right - frac, right))
            theta_ = np.expand_dims(theta_[ind][:, ind], axis=0)
            dist_ = np.expand_dims(dist_[ind][:, ind], axis=0)
            omega_ = np.expand_dims(omega_[ind][:, ind], axis=0)
            phi_ = np.expand_dims(phi_[ind][:, ind], axis=0)
            a3m = np.expand_dims(a3m[:, ind], axis=0)
            rrcs = np.expand_dims(rrcs[ind], axis=0)
            if temp_npz is not None:
                theta_temp_ = theta_temp_[:, ind][:, :, ind]
                dist_temp_ = dist_temp_[:, ind][:, :, ind]
                omega_temp_ = omega_temp_[:, ind][:, :, ind]
                phi_temp_ = phi_temp_[:, ind][:, :, ind]

        else:
            len_ = np.random.choice(range(min(cutoff, 150), cutoff))
            start_ = np.random.choice(a3m.shape[1] - len_)
            theta_ = np.array([theta_[start_:start_ + len_, start_:start_ + len_]])
            phi_ = np.array([phi_[start_:start_ + len_, start_:start_ + len_]])
            dist_ = np.array([dist_[start_:start_ + len_, start_:start_ + len_]])
            omega_ = np.array([omega_[start_:start_ + len_, start_:start_ + len_]])
            a3m = np.array([a3m[:, start_:start_ + len_]])
            rrcs = np.array([rrcs[start_:start_ + len_]])

            if temp_npz is not None:
                theta_temp_ = theta_temp_[:, start_:start_ + len_, start_:start_ + len_]
                phi_temp_ = phi_temp_[:, start_:start_ + len_, start_:start_ + len_]
                dist_temp_ = dist_temp_[:, start_:start_ + len_, start_:start_ + len_]
                omega_temp_ = omega_temp_[:, start_:start_ + len_, start_:start_ + len_]
    else:
        theta_ = np.array([theta_])
        phi_ = np.array([phi_])
        dist_ = np.array([dist_])
        omega_ = np.array([omega_])

        a3m = np.array([a3m])
        rrcs = np.array([rrcs])

    out = [a3m, rrcs, theta_, phi_, dist_, omega_]
    if temp_npz is not None:
        out += [theta_temp_, phi_temp_, dist_temp_, omega_temp_]

    return out


def binning(dist_, omega_, theta_, phi_):
    bins = np.linspace(2, 20, 36 + 1)
    bins180 = np.linspace(0.0, np.pi, 13)
    bins360 = np.linspace(-np.pi, np.pi, 25)

    # bin distance matrix
    dbin = np.digitize(dist_, bins).astype(np.uint8)
    dbin[dbin > 36] = 0

    # bin omega
    obin = np.digitize(omega_, bins360).astype(np.uint8)
    obin[dbin == 0] = 0

    # bin theta
    tbin = np.digitize(theta_, bins360).astype(np.uint8)
    tbin[dbin == 0] = 0

    # bin phi
    pbin = np.digitize(phi_, bins180).astype(np.uint8)
    pbin[dbin == 0] = 0
    return dbin, obin, tbin, pbin


def lr_schedule(epoch):
    if epoch < 10: return 1
    elif epoch < 15: return .7
    elif epoch < 20: return .4
    elif epoch < 25: return .2
    elif epoch < 30: return .1
    elif epoch < 35: return .05
    elif epoch < 40: return .02
    else: return .01



def get_vars(model_name, conv_drop=4, instancenorm_drop=0, conv_keep=None, instancenorm_keep=None):
    """ get trainable variables in a pre-trained model"""
    if conv_keep: conv_drop = 0
    if instancenorm_keep: instancenorm_drop = 0
    checkpoint_path = os.path.join(model_name)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    in_ind, conv_ind = [], []
    var_lst = list(var_to_shape_map.keys())
    for v in var_lst:
        if v.startswith('InstanceNorm_'):
            in_ind.append(int(re.search(r'InstanceNorm_(\d{1,3})/', v).groups()[0]))
        if v.startswith('conv2d_'):
            conv_ind.append(int(re.search(r'conv2d_(\d{1,3})/', v).groups()[0]))
    in_ind.sort()
    conv_ind.sort()

    def need_to_drop(v):
        if v.startswith('InstanceNorm_'):
            if int(re.search(r'InstanceNorm_(\d{1,3})/', v).groups()[0]) > in_ind[-1] - instancenorm_drop:
                # print(f'remove: {v}')
                return True
        if v.startswith('conv2d_'):
            if int(re.search(r'conv2d_(\d{1,3})/', v).groups()[0]) > conv_ind[-1] - conv_drop:
                # print(f'remove: {v}')
                return True
        return False

    def need_to_keep(v):
        if v.startswith('InstanceNorm/') or v.startswith('conv2d/'):
            return True
        if v.startswith('InstanceNorm_'):
            if int(re.search(r'InstanceNorm_(\d{1,3})/', v).groups()[0]) < instancenorm_keep:
                # print(f'keep: {v}')
                return True
        if v.startswith('conv2d_'):
            if int(re.search(r'conv2d_(\d{1,3})/', v).groups()[0]) < conv_keep:
                # print(f'keep: {v}')
                return True
        return False

    var_lst = [v for v in var_lst if not need_to_drop(v)]
    if conv_keep:
        var_lst = [v for v in var_lst if need_to_keep(v)]
    return var_lst
