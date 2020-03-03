"""Module with CIFAR100-like datasets
"""

import torchvision.datasets

from vari.datasets import MNISTContinuous


class CIFAR10Continuous(MNISTContinuous):
    """CIFAR10 dataset including filtering and concatenation of train and test sets. 
    See MNISTContinuous.
    """
    _data_source = torchvision.datasets.CIFAR10

    def __init__(self, split='train', exclude_labels=None, gamma=0.0, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, gamma=gamma, seed=seed,
                         root=root, transform=transform, target_transform=target_transform, download=download)


class CIFAR100Continuous(MNISTContinuous):
    """CIFAR100 dataset including filtering and concatenation of train and test sets. 
    See MNISTContinuous.
    """
    _data_source = torchvision.datasets.CIFAR100

    def __init__(self, split='train', exclude_labels=None, gamma=0.0, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, gamma=gamma, seed=seed,
                         root=root, transform=transform, target_transform=target_transform, download=download)


class SVHNContinuous(MNISTContinuous):
    """SVHN dataset including filtering and concatenation of train and test sets. 
    See MNISTContinuous.
    """
    _data_source = torchvision.datasets.SVHN

    def __init__(self, split='train', exclude_labels=None, gamma=0.0, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, gamma=gamma, seed=seed,
                         root=root, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, idx):
        example, label = super(SVHNContinuous, self).__getitem__(idx)
        if example.ndim == 3:
            example = example.transpose(1, 2, 0)
        elif example.ndim == 4:
            example = example.transpose(0, 2, 3, 1)
        else:
            raise ValueError
        return example, label
