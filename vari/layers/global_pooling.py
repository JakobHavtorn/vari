import torch.nn as nn
import torch.nn.functional as F


class _GlobalMaxPoolNd(nn.Module):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        super().__init__()
        
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=x.shape[-self._n_dims:]).squeeze()


class GlobalMaxPool1d(_GlobalMaxPoolNd):
    """Module that applies max pooling globally by have a kernel of the same size as the input"""
    def __init__(self):
        super().__init__(n_dims=1)


class GlobalMaxPool2d(_GlobalMaxPoolNd):
    """Module that applies max pooling globally by have a kernel of the same size as the input"""
    def __init__(self):
        super().__init__(n_dims=2)


class GlobalMaxPool1d(_GlobalMaxPoolNd):
    """Module that applies max pooling globally by have a kernel of the same size as the input"""
    def __init__(self):
        super().__init__(n_dims=3)
