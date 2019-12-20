import numpy as np

from torch.utils.data import Dataset


def make_data_spiral(n_samples=1000, shuffle=True, noise=.05, rotation=0, start_radius=np.pi, rounds=1,
                     normalizing_constant=2*np.pi):
    rot_mat = np.array([[np.cos(rotation),-np.sin(rotation)],
                        [np.sin(rotation), np.cos(rotation)]])
    
    #theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    theta = np.linspace(0, rounds * 2 * np.pi, n_samples)

    r_a = -theta - start_radius
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + noise * np.random.randn(n_samples, 2)
    x_a = np.dot(x_a, rot_mat)

    r_b = theta + start_radius
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + noise * np.random.randn(n_samples, 2)
    x_b = np.dot(x_b, rot_mat)
    
    res_a = np.append(x_a, np.zeros((n_samples, 1)), axis=1)
    res_b = np.append(x_b, np.ones((n_samples, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    values = res[:, :2] / normalizing_constant
    labels = np.identity(res[:, 2].astype(int).max() + 1)[res[:, 2].astype(int)]
    return values.astype(np.float32), labels.astype(np.int32)


class Spirals(Dataset):
    def __init__(self, n_samples=1000, noise=0.05, rotation=0, start_radius=np.pi, rounds=1):
        examples, labels = make_data_spiral(n_samples=n_samples, noise=noise,
                                            rotation=rotation, start_radius=start_radius, rounds=rounds)
        self.examples = examples
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.examples[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.examples)

