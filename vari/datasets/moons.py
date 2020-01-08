import numpy as np

from sklearn.datasets import make_moons
from torch.utils.data import Dataset


def make_data_moons(n_samples=1000, shuffle=True, noise=.05, seed=0):
    x, y = make_moons(n_samples, shuffle, noise, random_state=seed)
    x[:, 1][y == 0] = x[:, 1][y == 0]
    y = np.identity(y.max() + 1)[y]
    return x.astype(np.float32), y.astype(np.float32)


class Moons(Dataset):
    def __init__(self, n_samples=1000, noise=0.05, seed=0):
        examples, labels = make_data_moons(n_samples=n_samples, noise=noise, seed=seed)
        self.examples = examples
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.examples)