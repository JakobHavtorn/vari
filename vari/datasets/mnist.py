"""Module with MNIST-like datasets

MNISTBinarized and MNISTReal form the parent classes of the remaining datasets.
"""

import os
import urllib

import numpy as np

import torchvision.datasets

from torch.utils.data import Dataset


def onehot_encode(array, max_label=None):
    max_label = np.max(array) + 1 if max_label is None else max_label
    return np.eye(max_label)[array]


class MNISTBinarized(Dataset):
    """MNIST dataset including filtering and concationation of train and test sets.
    
    Serves binarized values in {0, 1}.
    
    The non-deterministic preprocessing, which consists of binarization using the normalized pixel values [0, 1] as
    binomial probabilities, can either be done once ahead of training `preprocess=='static'` or done anew for each
    example while training, `preprocess=='dynamic'`. Setting the `seed` will ensure reproducibility in either case.
    The third option for `preprocess` is `deterministic` which applies a different, deterministic preprocessing that
    binarizes images at a pixel value of 0.5.
    
    If preprocess is 'static', the noise added to the dataset at different values of `split` is NOT the same per
    example. I.e. examples from the training set are modified differently when served with `split=='train'` and 
    `split=='join'`.
    
    Args:
        split (str): Whether to serve the 'train' or 'test' sets or 'join' the two to a single set.
        exclude_labels (list): List of integer labels to exclude from the dataset.
        preprocess (bool): If 'static', performs preprocessing before training.
                           If 'deterministic' preprocessing is done deterministically before training.
                           If 'dynamic' preprocessing is done on the fly. 
    """

    _data_source = torchvision.datasets.MNIST
    _repr_attributes = ['split', 'exclude_labels', 'preprocess', 'seed', 'root']

    def __init__(self, split='train', exclude_labels=None, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__()
        
        assert preprocess in ['dynamic', 'static', 'deterministic']
        assert split in ['train', 'test', 'join']

        self.split = split
        self.exclude_labels = [] if exclude_labels is None else exclude_labels
        self.preprocess = preprocess
        self.seed = seed
        self.root = root

        np.random.seed(seed)
        data_train = self._data_source(root=root, train=True, transform=transform,
                                       target_transform=target_transform, download=download)
        data_test = self._data_source(root=root, train=False, transform=transform,
                                      target_transform=target_transform, download=download)

        if split != 'join':
            if split == 'train':
                self.examples = np.array(data_train.data)
                self.labels = onehot_encode(np.array(data_train.targets))
            else:
                self.examples = np.array(data_test.data)
                self.labels = onehot_encode(np.array(data_test.targets))
        else:
            self.examples = np.concatenate([data_train.data, data_test.data])
            self.labels = onehot_encode(np.concatenate([data_train.targets, data_test.targets]))

        if preprocess in ['static', 'deterministic']:
            self.examples = self.scale(self.examples)
            self.examples = self.warp(self.examples)

        self.examples, self.labels = self.filter(self.examples, self.labels, exclude_labels)

    def __getitem__(self, idx):
        example = self.examples[idx].astype(np.float32)
        if self.preprocess == 'dynamic':
            example = self.warp(self.scale(example))
        return example.astype(np.float32), self.labels[idx].astype(np.float32)
    
    @staticmethod
    def filter(examples, labels, exclude_labels):
        """Filter the dataset and return examples and labels without the excluded labels"""
        rm_ids = np.isin(labels.argmax(axis=1), exclude_labels)
        labels = labels[~rm_ids]
        examples = examples[~rm_ids]
        return examples, labels

    def scale(self, examples):
        """Standard scaling of MNIST.

        Adds uniform [0, 1] noise to the integer pixel values between 0 and 255 and then divides by 256.
        This results in values in [0, 1].
        """
        examples = examples.astype(np.float64)
        examples /= 255  # /= examples.max()
        return examples

    def warp(self, examples):
        if self.preprocess == 'deterministic':
            idx = examples >= 0.5
            examples[idx] = 1.0
            examples[~idx] = 0.0
            return examples
        return np.random.binomial(1, examples)  # Binary sampling with normalized pixel values as probabilities

    def __len__(self):
        return self.examples.shape[0]

    def __repr__(self):
        s = f'{self.__class__.__name__}('
        s += ', '.join([f'{attr}={getattr(self, attr)}' for attr in self._repr_attributes])
        return s + ')'


class MNISTReal(MNISTBinarized):
    """MNIST dataset including filtering and concationation of train and test sets.
    
    Serves real values in [0, 1].
    
    The non-deterministic preprocessing, which consists of the addition of uniform noise to the raw pixel values and
    then interpolable binarization through setting gamma, can be done once ahead of training `preprocess=='static'` or
    done anew for each example while training, `preprocess=='dynamic'`. Setting the `seed` will ensure reproducibility
    in either case. The third option for `preprocess` is `deterministic` which applies a different, deterministic
    preprocessing that binarizes images at a pixel value of 0.5.

    If preprocess is 'static', the noise added to the dataset at different values of `split` is NOT the same per
    example. I.e. examples from the training set are modified differently when served with `split=='train'` and 
    `split=='join'`.
    
    Args:
        split (str): Whether to serve the 'train' or 'test' sets or 'join' the two to a single set.
        exclude_labels (list): List of integer labels to exclude from the dataset.
        gamma (float): Value in range [-0.5, 0.5] which interpolates between binarization (-0.5) and degrading (0.5)
        preprocess (bool): If 'static', performs preprocessing before training.
                           If 'deterministic' preprocessing is done deterministically before training.
                           If 'dynamic' preprocessing is done on the fly. 
    """
    _data_source = torchvision.datasets.MNIST
    _repr_attributes = MNISTBinarized._repr_attributes + ['gamma']

    def __init__(self, split='train', exclude_labels=None, preprocess='dynamic', gamma=0.0, seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        assert gamma <= 0.5 and gamma >= -0.5, 'gamma must be in [-0.5, 0.5]'
        self.gamma = gamma
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, seed=seed, root=root,
                         transform=transform, target_transform=target_transform, download=download)

    def scale(self, examples):
        """Standard preprocessing of MNIST.

        Adds uniform [0, 1] noise to the integer pixel values between 0 and 255 and then divides by 256.
        This results in continuous values in [0, 1].
        """
        examples = examples.astype(np.float32)
        if self.preprocess != 'deterministic':
            noise_matrix = np.random.rand(*examples.shape)
            examples += noise_matrix
            examples /= 256
        else:
            examples /= 255
        return examples

    def warp(self, examples):
        """
        Warping function that warps pixels from fully binarized {0,1} for gamma=0.5 to fully degraded (=0.5) for 
        gamma=-0.5.
        
        Implemented as in [1].
        
        [1] The continuous Bernoulli: fixing a pervasive error in variational autoencoders
            http://arxiv.org/abs/1907.06845
        """
        assert self.gamma <= 0.5 and self.gamma >= -0.5
        if self.gamma == 0:
            return examples
        if self.gamma == -0.5:
            idx = examples < 0.5
            examples[idx] = 0
            examples[~idx] = 1
            return examples
        if self.gamma < 0:
            return np.clip((examples + self.gamma) / (1 + 2 * self.gamma), 0, 1)
        return self.gamma + (1 - 2 * self.gamma) * examples
    

class MNISTBinarizedLarochelle(Dataset):
    def __init__(self, split, exclude_labels=None, root='torch_data/', transform=None, target_transform=None, download=True):
        self.split = split
        self.exclude_labels = [] if exclude_labels is None else exclude_labels
        self.download(root)
        self.examples, self.labels = self.load(root, split)

    def __getitem__(self, idx):
        return self.examples[idx].astype(np.float32), self.labels[idx].astype(np.float32)

    @staticmethod
    def load(root, split):
        if split in ['train', 'valid', 'test']:
            examples = np.load(os.path.join(root, 'MNISTBinarized', f'{split}.npy'))
        elif split == 'join':
            train = np.load(os.path.join(root, f'train.npy'))
            valid = np.load(os.path.join(root, f'valid.npy'))
            test = np.load(os.path.join(root, f'test.npy'))
            examples = np.concatenate([train, valid, test])
        return examples, np.zeros(shape=(examples.shape[0],))

    @staticmethod
    def download(root):
        """
        Download the binzarized MNIST dataset if it is not present.
        :return: The train, test and validation set.
        """
        split_urls = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_data_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_data_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_data_test.amat"
        }
        if not os.path.exists(os.path.join(root, 'MNISTBinarized')):
            os.makedirs(os.path.join(root, 'MNISTBinarized'))

        import IPython
        IPython.embed()

        for split in split_urls.keys():
            npy_file = root + f'/MNISTBinarized/{split}.npy'
            if os.path.exists(npy_file):
                continue
            print(f'Downloading MNISTBinarized {split} data...')
            # TODO BinarizedMNIST Find the associated labels
            data = np.loadtxt(urllib.request.urlretrieve(split_urls[split])[0])
            data = data.reshape(data.shape[0], 28, 28)
            np.save(npy_file, data, allow_pickle=False)
            
    def __len__(self):
        return self.examples.shape[0]

    def __repr__(self):
        s = f'MNISTBinarizedLarochelle('
        s += ', '.join([f'{attr}={getattr(self, attr)}' for attr in ['split', 'exclude_labels']])
        return s + ')'


class FashionMNISTReal(MNISTReal):
    """FashionMNIST dataset including filtering and concationation of train and test sets. 
    See MNISTReal.
    """

    _data_source = torchvision.datasets.FashionMNIST

    def __init__(self, split='train', exclude_labels=None, gamma=0.0, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, gamma=gamma, seed=seed,
                         root=root, transform=transform, target_transform=target_transform, download=download)
        

class FashionMNISTBinarized(MNISTBinarized):
    """FashionMNIST dataset including filtering and concationation of train and test sets.
    See MNISTBinarized.
    """
    _data_source = torchvision.datasets.FashionMNIST

    def __init__(self, split='train', exclude_labels=None, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, seed=seed, root=root,
                         transform=transform, target_transform=target_transform, download=download)



class CIFAR10Real(MNISTReal):
    _data_source = torchvision.datasets.CIFAR10
    
    def __init__(self, split='train', exclude_labels=None, preprocess='dynamic', seed=0, root='torch_data/',
                 transform=None, target_transform=None, download=True):
        super().__init__(split=split, exclude_labels=exclude_labels, preprocess=preprocess, seed=seed, root=root,
                         transform=transform, target_transform=target_transform, download=download)
    