from itertools import repeat

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from vari.utilities import log_sum_exp, enumerate_discrete
from vari.inference.distributions import log_standard_categorical


class ImportanceWeightedSampler():
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class FreeNatsCooldown():
    """Linear deterministic warm-up as described in [Sønderby 2016]. 
    """
    def __init__(self, constant_epochs=200, cooldown_epochs=200, start_val=0.2, end_val=None):
        self.constant_epochs = constant_epochs
        self.cooldown_epochs = cooldown_epochs
        self.start_val = start_val
        self.end_val = start_val if constant_epochs == cooldown_epochs == 0 else end_val  # Start val if zero duration
        end_val = 0 if end_val is None else end_val
        self.values = np.concatenate([
            np.array([start_val] * constant_epochs),  # [start_val, start_val, ..., start_val]
            np.linspace(start_val, end_val, cooldown_epochs)  # [start_val, ..., end_val]
        ])
        self.i_epoch = -1

    @property
    def is_done(self):
        return not self.i_epoch < len(self.values)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.i_epoch += 1
        if self.is_done:
            return self.end_val
        return self.values[self.i_epoch]


class DeterministicWarmup():
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=200, t_max=1, t_start=0):
        self.n = n
        self.t_max = t_max
        self.t = t_start if n != 0 else t_max
        self.inc = 1 / n if n != 0 else 0

    @property
    def is_done(self):
        return self.t >= self.t_max

    def __iter__(self):
        return self

    def __next__(self):
        self.t += self.inc
        if self.t >= self.t_max:
            return self.t_max
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)

    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        # Increase sampling dimension
        xs = self.sampler.resample(xs)
        ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        return torch.mean(U)