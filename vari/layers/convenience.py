import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    """Module that flattens an input of dimensions [B, D1, D2, ...] to [B, D1 * D2 * ...]"""
    def __init__(self, n_batch_dims=1):
        self.n_batch_dims = n_batch_dims
        super().__init__()

    def forward(self, x):
        return x.view(*x.shape[:self.n_batch_dims], -1)

    def extra_repr(self):
        return f'n_batch_dims={self.n_batch_dims}'


class View(nn.Module):
    """Module that returns a view of an input"""
    def __init__(self, n_batch_dims=1, shape=(-1, 1, 1)):
        self.shape = shape
        self.n_batch_dims = n_batch_dims
        super().__init__()

    def forward(self, x):
        return x.view(*x.shape[:self.n_batch_dims], *self.shape)

    def extra_repr(self):
        return f'n_batch_dims={self.n_batch_dims}, shape={self.shape}'


class AddConstant(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f'AddConstant({self.constant})'


class Clamp(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min if min is not None else -float('inf')
        self.max = max if max is not None else float('inf')

    def forward(self, tensor):
        return tensor.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f'Clamp({self.min, self.max})'
