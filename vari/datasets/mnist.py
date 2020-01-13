import numpy as np

from torch.utils.data import Dataset
import torchvision.datasets


def onehot_encode(array, max_label=None):
    max_label = np.max(array) + 1 if max_label is None else max_label
    return np.eye(max_label)[array]


"""
We preprocess MNIST by following the standard procedure of adding uniform [0, 1] noise to the integer pixel values
 between 0 and 255 and then dividing by 256, resulting in values in [0, 1]. 
 
 For all our MNIST experiments, we use a 
 latent dimension of 20, an encoder with two hidden layers with 500 units each, with ReLU nonlinearities, followed by 
 a dropout layer (with parameter 0.9). The output layer of the encoder has no nonlinearity for the mean and a softplus 
 nonlinearity for the standard deviation. 
 
 The decoder also has two hidden layers with 500 units, ReLU nonlinearities 
 and dropout, as does the classiﬁer we used to compute the inception score (which has a softmax nonlinearity). 
 
 The decoder has softplus nonlinearities to enforce nonnegativity (Gaussian standard deviation and beta parameters), 
 sigmoid to enforce values in (0, 1) (continuous Bernoulli, Bernoulli and Gaussian mean).
"""
    

class MNIST(Dataset):
    """MNIST dataset including filtering and concationation of train and test sets
    
    Args:
        split (str): Whether to serve the 'train' ir 'test' sets or 'join' the two to a single set.
        exclude_labels (list): List of integer labels to exclude from the dataset.
        gamma (float): Value in range [-0.5, 0.5] which interpolates between binarization (-0.5) and degrading (0.5)
    """

    def __init__(self, split='train', exclude_labels=None, gamma=0.0, seed=0, root='torch_data/', transform=None, target_transform=None, download=True):
        self.gamma = gamma
        self.split = split
        self.seed = seed
        self.exclude_labels = [] if exclude_labels is None else exclude_labels
        
        mnist_train = torchvision.datasets.MNIST(root=root, train=True, transform=transform,
                                            target_transform=target_transform, download=download)
        mnist_test = torchvision.datasets.MNIST(root=root, train=False, transform=transform,
                                            target_transform=target_transform, download=download)

        np.random.seed(seed)
        # self.noise_matrix = np.random.rand(mnist_train.data.shape[0] + mnist_test.data.shape[0],
        #                                    *mnist_train.data.shape[1:])

        if split != 'join':
            if split == 'train':
                self.examples = np.array(mnist_train.data)
                self.labels = onehot_encode(np.array(mnist_train.targets))
            else:
                self.examples = np.array(mnist_test.data)
                self.labels = onehot_encode(np.array(mnist_test.targets))
        else:
            self.examples = np.concatenate([mnist_train.data, mnist_test.data])
            self.labels = onehot_encode(np.concatenate([mnist_train.targets, mnist_test.targets]))

        # self.examples = self.preprocess(self.examples, seed=seed)
        self.examples, self.labels = self.filter(self.examples, self.labels, exclude_labels)
        self.examples = self.warp_pixels(self.examples, gamma)

    def filter(self, examples, labels, exclude_labels):
        """Filter the dataset and return examples and labels without the excluded labels"""
        rm_ids = np.isin(labels.argmax(axis=1), exclude_labels)
        labels = labels[~rm_ids]
        examples = examples[~rm_ids]
        return examples, labels
    
    def preprocess(self, examples, seed):
        """Standard preprocessing of MNIST.

        Adding uniform [0, 1] noise to the integer pixel values between 0 and 255 and then dividing by 256.
        This results in values in [0, 1]
        """
        examples = examples.astype(np.float64)
        examples += self.noise_matrix[:examples.shape[0]]
        examples /= 256
        return examples

    def warp_pixels(self, examples, gamma):
        """
        Warping function that warps pixels from fully binarized {0,1} for gamma=0.5 to fully degraded (=0.5) for 
        gamma=-0.5.
        
        Implemented as in [1].
        
        [1] The continuous Bernoulli: ﬁxing a pervasive error in variational autoencoders
            http://arxiv.org/abs/1907.06845
        """
        assert gamma <= 0.5 and gamma >= -0.5
        if gamma == 0:
            return examples
        if gamma == -0.5:
            idx = examples < 0.5
            examples[idx] = 0
            examples[~idx] = 1
            return examples
        if gamma < 0:
            return np.clip((examples + gamma) / (1 + 2 * gamma), 0, 1)
        return gamma + (1 - 2 * gamma) * examples

    def __getitem__(self, idx):
        return self.examples[idx].astype(np.float32), self.labels[idx].astype(np.float32)

    def __len__(self):
        return self.examples.shape[0]
    
    def __repr__(self):
        s = f'MNIST('
        s += ', '.join([f'{attr}={getattr(self, attr)}' for attr in ['split', 'exclude_labels', 'gamma']])
        return s + ')'
