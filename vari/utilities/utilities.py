import os
import math

from collections import OrderedDict
from itertools import repeat

import torch
import torch.nn as nn

from torch._six import container_abcs

from torch.autograd import Variable


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def get_device(idx=None):
    if 'CUDA_VISIBLE_DEVICES' in os.environ and torch.cuda.is_available():
        global_device_indices = [int(i.rstrip().lstrip()) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        local_device_indices = list(range(len(global_device_indices)))
        if idx is None:
            return torch.device('cuda:0')
        return torch.device(f'cuda:{local_device_indices[idx]}')
    return torch.device('cpu')


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.
    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def onehot_encode(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def log_sum_exp(tensor, axis=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param axis: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    maximum, _ = torch.max(tensor, axis=axis, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - maximum), axis=axis, keepdim=True) + 1e-8) + maximum


def activation_gain(activation):
    """Return the gain associated with the given activation according to 
    
    Args:
        activation ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    if activation is None:
        return 1
    elif isinstance(activation, nn.LeakyReLU) or isinstance(activation, nn.ELU):
        return nn.init.calculate_gain('leaky_relu', param=None)
    name = activation.__class__.__name__.lower()
    return nn.init.calculate_gain(name, param=None)


def compute_convolution_output_dimensions(i, k, s=None, p=None, transposed=False):
    """Compute the output dimensions for a convolution.
    
    Args:
        i (tuple): Input shape
        p (tuple): Padding shape
        k (tuple): Kernel shape
        s (tuple): Stride shape
    
    Returns:
        tuple: Output shape
    """
    def regular_conv(_i, _k, _s, _p):
        return math.floor((_i + 2 * _p - _k) / _s) + 1
    
    def transposed_conv(_i, _k, _s, _p):
        """
        A convolution described by k, s and p and whose input size i is such that i+2p−k is a multiple of s has an
        associated transposed convolution described by î , k' = k, s' = 1 and p' = k − p − 1, where î  is the
        size of the stretched input obtained by adding s − 1 zeros between each input unit, and its output size is
        """
        return _s * (_i - 1) + _k - 2 * _p
    
    i = (i,) if isinstance(i, int) else i
    k = (k,) * len(i) if isinstance(k, int) else k

    s = s if s is not None else [1] * len(i)
    s = (s,) * len(i) if isinstance(s, int) else s
    p = p if p is not None else [0] * len(i)
    p = (p,) * len(i) if isinstance(p, int) else p
    
    if not transposed:
        return [regular_conv(_i, _k, _s, _p) for _i, _k, _s, _p in zip(i, k, s, p)]
    return [transposed_conv(_i, _k, _s, _p) for _i, _k, _s, _p in zip(i, k, s, p)]


def compute_output_padding(i, k, s=None, p=None):
    """
    Compute the amount of zero padding to add to the bottom and right edges of the input of a tranposed convolution
    in order to select among the multiple cases that all lead to the same i'.
    
    This returns 0 whenever s = 0 since the result is computed modulo s.
    
    Reference: Dumoulin. 2018. A guide to convolution arithmetic for deep learning. Relationship 14.
    
    Args:
        i (tuple): Input shape
        p (tuple): Padding shape
        k (tuple): Kernel shape
        s (tuple): Stride shape
    
    Returns:
        int: Amount of zero padding to be added to input (`output_padding` argument of nn.ConvNdTranspose)
    """
    def _compute_output_padding(_i, _k, _s, _p):
        return (_i + 2 * _p - _k) % _s
    
    if s is None:
        return 0

    i = (i,) if isinstance(i, int) else i
    k = (k,) * len(i) if isinstance(k, int) else k
    s = (s,) * len(i) if isinstance(s, int) else s
    p = p if p is not None else [0] * len(i)
    p = (p,) * len(i) if isinstance(p, int) else p

    return [_compute_output_padding(_i, _k, _s, _p) for _i, _k, _s, _p in zip(i, k, s, p)]


def remove_outliers(*data, thresh=3.5):
    out = []
    ids = []
    for d in data:
        idx = is_outlier(d, thresh=thresh)
        out.append(d[~idx])
        ids.append(idx)
    return (out, ids) if len(out) > 1 else (out[0], ids[0])


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = torch.median(points, axis=0).values
    diff = torch.sum((points - median)**2, axis=-1)
    diff = torch.sqrt(diff)
    med_abs_deviation = torch.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def concatenate_on_keys(list_of_dict):
    dict_of_array = OrderedDict()
    for kl in list_of_dict:
        for k, v in kl.items():
            if k not in dict_of_array:
                dict_of_array[k] = v
            else:
                dict_of_array[k] = torch.cat([dict_of_array[k], v])
    return dict_of_array


def compute_dataset_elbo(model, loader, device='cpu', **kwargs):
    all_elbo = []
    all_likelihood = []
    all_kl = []
    all_kls = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=False)
            elbo, like, kl = model.elbo(x.view(x.shape[0], *model.in_shape), **kwargs)
            all_elbo.append(elbo)
            all_likelihood.append(like)
            all_kl.append(kl)
            all_kls.append(model.kl_divergences)
    return torch.cat(all_elbo), torch.cat(all_likelihood), torch.cat(all_kl), concatenate_on_keys(all_kls)


def encode_dataset(model, loader, device='cpu', **kwargs):
    distributions
    latents = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=False)
            lat = model.encode(x.view(x.shape[0], *model.in_shape), **kwargs)
            latents.append(lat)
    return concatenate_on_keys(latents)
