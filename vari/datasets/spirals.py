import numpy as np

from torch.utils.data import Dataset


def make_data_spiral(n_samples=1000, shuffle=True, noise=.05, radius=1):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + noise * np.random.randn(n_samples, 2)

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + noise * np.random.randn(n_samples, 2)

    res_a = np.append(x_a, np.zeros((n_samples, 1)), axis=1)
    res_b = np.append(x_b, np.ones((n_samples, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    values = radius * res[:, :2] / np.abs(res[:, :2]).max()
    labels = np.identity(res[:, 2].astype(int).max() + 1)[res[:, 2].astype(int)]
    return values.astype(np.float32), labels.astype(np.float32)


class Spirals(Dataset):
    def __init__(self, n_samples=1000, noise=0.05, radius=1):
        examples, labels = make_data_spiral(n_samples=n_samples, noise=noise, radius=radius)
        self.examples = examples
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.examples)

