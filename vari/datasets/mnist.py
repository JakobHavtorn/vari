import numpy as np

from torch.utils.data import Dataset
import torchvision.datasets


def onehot_encode(array, max_label=None):
    max_label = np.max(array) + 1 if max_label is None else max_label
    return np.eye(max_label)[array]
    

class MNIST(Dataset):
    """MNIST dataset including filtering and concationation of train and test sets
    
    Args:
        split (str): Whether to serve the 'train' ir 'test' sets or 'join' the two to a single set.
        exclude_labels (list): List of integer labels to exclude from the dataset.
    """
    def __init__(self, split='join', exclude_labels=None, root='torch_data/', transform=None, target_transform=None, download=True):
        self.split = split
        self.exclude_labels = [] if exclude_labels is None else exclude_labels
        
        if split != 'join':
            train = split == 'train'
            mnist = torchvision.datasets.MNIST(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
            self.examples = mnist.data
            self.labels = onehot_encode(mnist.targets)
        else:
            mnist1 = torchvision.datasets.MNIST(root=root, train=True, transform=transform, target_transform=target_transform, download=download)
            mnist2 = torchvision.datasets.MNIST(root=root, train=False, transform=transform, target_transform=target_transform, download=download)
            self.examples = np.concatenate([mnist1.data, mnist2.data])
            self.labels = onehot_encode(np.concatenate([mnist1.targets, mnist2.targets]))
            
        self.examples, self.labels = self.filter(exclude_labels)
        
    def filter(self, exclude_labels):
        """Filter the dataset and return examples and labels without the excluded labels"""
        rm_ids = np.isin(self.labels.argmax(axis=1), exclude_labels)
        labels = self.labels[~rm_ids]
        examples = self.examples[~rm_ids]
        return examples, labels

    def __getitem__(self, idx):
        example = self.examples[idx] / 255
        return example.astype(np.float32), self.labels[idx].astype(np.float32)

    def __len__(self):
        return self.examples.shape[0]