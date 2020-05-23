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
        
    def scale(self, examples):
        # SVHN comes as [C, H, W] instead of [H, W, C] which is the convention we use so we tranpose
        if examples.ndim == 3:  # Not batched
            examples = examples.transpose(1, 2, 0)
        elif examples.ndim == 4:  # Batched
            examples = examples.transpose(0, 2, 3, 1)
        else:
            raise ValueError
        return super().scale(examples)