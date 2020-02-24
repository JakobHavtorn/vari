import math

from numbers import Number

import torch

from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property
from torch.nn.functional import binary_cross_entropy_with_logits



class ContinuousBernoulli(ExponentialFamily):
    r"""
    Creates a Continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    arg_constraints = {'probs': constraints.unit_interval,
                       'logits': constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            self.probs, = broadcast_all(probs)
        else:
            is_scalar = isinstance(logits, Number)
            self.logits, = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super(ContinuousBernoulli, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        # Taylor approximation of x/(2*x-1) + 1/(2*arctanh(1-2*x))
        # https://www.wolframalpha.com/input/?i=taylor+series+of+x%2F%282*x-1%29+%2B+1%2F%282*tanh%5E%28-1%29%281-2*x%29%29++at+x+%3D+0.5
        x_x0 = self.probs - 0.5
        taylor = 0.5 + x_x0/3 + 16/45*x_x0**3 + 704/945*x_x0**5 + 27392/14175*x_x0**7 + 2610176/467775*x_x0**9
        return taylor

    @property
    def variance(self):
        # Taylor approximation of ((x-1)*x)/(1-2*x)^2 + 1/(2*arctanh(1-2*x))^2
        # https://www.wolframalpha.com/input/?i=taylor+series+of+%28%28x-1%29*x%29%2F%281-2*x%29%5E2+%2B+1%2F%282*tanh%5E%28-1%29%281-2*x%29%29%5E2++at+x+%3D+0.5
        x_x0 = self.probs - 0.5
        taylor = 1/12 - 1/15*x_x0**2 - 128/945*x_x0**4 - 4864/14175*x_x0**6 - 151552/155925*x_x0**8 - 1881116672/638512785*x_x0**10
        return taylor

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.mean.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        # Taylor approximation of log( (2*arctanh(1-2x)) / (1 - 2*x) ))
        # https://www.wolframalpha.com/input/?i=taylor+series+of+log%28+%282+*+arctanh%281-2*x%29%29+%2F+%281+-+2*x%29+%29+at+x%3D0.5
        # Taylor approximation of (2*arctanh(1-2x)) / (1 - 2*x)
        # https://www.wolframalpha.com/input/?i=taylor+series+of+%282+*+arctanh%281-2*x%29%29+%2F+%281+-+2*x%29+at+x%3D0.5
        x_x0 = self.probs - 0.5
        log_C = torch.log(2 + 8/3*(x_x0)**2 + 32/5*(x_x0)**4 + 128/7*(x_x0)**6 + 512/9*(x_x0**8 + 2048/11*(x_x0**10)))
        # log_C = math.log(2) + 4/3*x_x0**2 + 104/45*x_x0**4 + 16064/2835*x_x0**6 + 227264/14175*x_x0**8 + 22955008/467775*x_x0**10
        return -binary_cross_entropy_with_logits(logits, value, reduction='none') + log_C

    # def entropy(self):
    #     return binary_cross_entropy_with_logits(self.logits, self.probs, reduction='none')

    def enumerate_support(self, expand=True):
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self):
        return (torch.log(self.probs / (1 - self.probs)), )

    def _log_normalizer(self, x):
        return torch.log(1 + torch.exp(x))
    