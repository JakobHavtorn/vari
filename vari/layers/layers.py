import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

from torch.autograd import Variable

from vari.utilities import get_device
from vari.inference.distributions import log_gaussian, log_bernoulli, log_continuous_bernoulli


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x):
        return x


class AddConstant(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant
        
    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f'AddConstant({self.constant})'


class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        
    def forward(self, tensor):
        return tensor.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f'Clamp({self.min, self.max})'


class Distribution(nn.Module):
    pass


class GaussianLayer(Distribution):
    """Layer that parameterizes a Gaussian distribution by its mean and standard deviation.
    
    The layer outputs a torch.distributions.Independent wrapped torch.distribution.Normal parameterized by the mean and
    standard deviation.
    
    The mean is parameterized by a linear transformation of the input without nonlinearity and has range [-inf, inf].
    The standard deviation is parameterized by a linear transformation of the input followed by a softplus activation
    scaling it to [0, inf] and then a Clamping to fix it in the range [min_sd, max_sd] = [1e-8, 10] by default.
    """
    def __init__(self, in_features, out_features, min_sd=1e-8, max_sd=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.scale = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Softplus(),
            Clamp(min=min_sd, max=max_sd)
        )
        self.initialize()
    
    def initialize(self):
        # Gain is that of ReLU for scale since it's close to the SoftPlus
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.scale[0].weight, gain=gain)
        # For the mean there is no nonlinearity so we simply take gain to be 1
        nn.init.xavier_normal_(self.mu.weight, gain=1.)

    def get_prior(self, mu=None, scale=None):
        if mu is None and scale is None:
            return torch.distributions.Independent(torch.distributions.Normal(
                loc=torch.zeros(self.out_features).to(get_device()),
                scale=torch.ones(self.out_features).to(get_device())
            ), 1)
            # NOTE The below allows using analytical KL divergence directly without modifications to torch
            # return torch.distributions.MultivariateNormal(
            #     loc=torch.zeros(self.out_features).to(get_device()),
            #     covariance_matrix=torch.eye(self.out_features).to(get_device())
            # )
        return torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
        # NOTE The below allows using analytical KL divergence directly without modifications to torch
        # cov = torch.diag_embed(scale ** 2)
        # return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

    def forward(self, x):
        mu = self.mu(x)
        scale = self.scale(x)
        return torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
        # NOTE The below allows using analytical KL divergence directly without modifications to torch
        # cov = torch.diag_embed(scale ** 2)  # [B, D] B batches of D scales --> [B, D, D] B batches of diagonal DxD matrices
        # return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)


class GaussianFixedVarianceLayer(Distribution):
    """Layer that parameterizes a Gaussian distribution by its mean and a fixed standard deviation.
    
    The layer outputs a torch.distributions.Independent wrapped torch.distribution.Normal parameterized by the mean and
    standard deviation.
    
    The mean is parameterized by a linear transformation of the input without nonlinearity and has range [-inf, inf].
    The standard deviation is fixed at a constant value of `std` and is not learnable.
    
    The log-likelihod of the distributions resulting from this layer correspond to the MSE loss function.
    """
    def __init__(self, in_features, out_features, std=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.scale = std * torch.ones(out_features).to(get_device())
        self.initialize()
        
    def initialize(self):
        # For the mean there is no nonlinearity so we simply take gain to be 1
        nn.init.xavier_normal_(self.mu.weight, gain=1.)
        
    def get_prior(self):
        return torch.distributions.Independent(torch.distributions.Normal(mu, self.scale), 1)

    def forward(self, x):
        mu = self.mu(x)
        return torch.distributions.Independent(torch.distributions.Normal(mu, self.scale), 1)


class BernoulliLayer(Distribution):
    """Layer that parameterizes a Bernoulli distribution."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )
        self.initialize()
    
    def initialize(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_normal_(self.p[0].weight, gain=gain)

    def forward(self, x):
        p = self.p.forward(x)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=p), 1)


# class ContinuousBernoulli(torch.distributions.Bernoulli):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         logits, value = broadcast_all(self.logits, value)
#         return -binary_cross_entropy_with_logits(logits, value, reduction='none')
    
    
class ContinuousBernoulliLayer(Distribution):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        raise NotImplementedError()
        # TODO Need to sublcass torch.distributions.Bernoulli to change log_prob and entropy with the normalizing
        # constant
        p = self.p(x)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=p), 1)
    
    def log_likelihood(self, x, p):
        return log_continuous_bernoulli(x, p)
        

class GaussianMerge(Distribution):
    """
    Precision weighted merging of two Gaussian
    distributions.
    Merges information from z into the given
    mean and log variance and produces
    a sample from this new distribution.
    """
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, z, mu1, log_var1):
        raise NotImplementedError('Not yet adapted to GaussianLayer layer using Softplus to return standard deviation')
        # Calculate precision of each distribution (inverse variance)
        mu2 = self.mu(z)
        #log_var2 = F.softplus(self.log_var(z))
        log_var2 = self.log_var(z)
        precision1, precision2 = (1 / torch.exp(log_var1), 1 / torch.exp(log_var2))

        # Merge distributions into a single new distribution
        mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)

        var = 1 / (precision1 + precision2)
        log_var = torch.log(var + 1e-8)

        return self.reparametrize(mu, log_var), (mu, log_var)


class GumbelSoftmax(Distribution):
    """
    Layer that represents a sample from a categorical
    distribution. Enables sampling and stochastic
    backpropagation using the Gumbel-Softmax trick.
    """
    def __init__(self, in_features, out_features, n_distributions):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_distributions = n_distributions

        self.logits = nn.Linear(in_features, n_distributions * out_features)

    def forward(self, x, tau=1.0):
        logits = self.logits(x).view(-1, self.n_distributions)

        # variational distribution over categories
        softmax = F.softmax(logits, dim=-1) #q_y
        sample = self.reparametrize(logits, tau).view(-1, self.n_distributions, self.out_features)
        sample = torch.mean(sample, dim=1)

        return sample, softmax

    def reparametrize(self, logits, tau=1.0):
        epsilon = Variable(torch.rand(logits.size()), requires_grad=False)  # TODO is this supposed to be randn?

        if logits.is_cuda:
            epsilon = epsilon.cuda()

        # Gumbel distributed noise
        gumbel = -torch.log(-torch.log(epsilon+1e-8)+1e-8)
        # Softmax as a continuous approximation of argmax
        y = F.softmax((logits + gumbel)/tau, dim=1)
        return y
