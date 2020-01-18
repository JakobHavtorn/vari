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


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Distribution(nn.Module):
    prior = None


class GaussianSample(Distribution):
    """
    Layer that parameterizes a Gaussian distribution.
    """
    def __init__(self, in_features, out_features, scale_as='std'):
        super().__init__()
        assert scale_as in ['std', 'log_var']
        self.in_features = in_features
        self.out_features = out_features
        self.scale_as = scale_as

        self.mu = nn.Linear(in_features, out_features)
        scale = [nn.Linear(in_features, out_features)]
        if scale_as == 'std':
            scale.append(nn.Softplus())
            scale.append(Lambda(lambda x: x + 1e-8))
        self.scale = nn.Sequential(*scale)

    def get_prior(self, mu=None, scale=None):
        if mu is None and scale is None:
            return torch.distributions.Independent(torch.distributions.Normal(
                torch.zeros(self.out_features).to(get_device()),
                torch.ones(self.out_features).to(get_device())
            ), 1)
            # return torch.distributions.MultivariateNormal(
            #     loc=torch.zeros(self.out_features).to(get_device()),
            #     covariance_matrix=torch.eye(self.out_features).to(get_device())
            # )
        return torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
        # cov = torch.diag_embed(scale ** 2)
        # return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
        

    def forward(self, x):
        mu = self.mu(x)
        scale = self.scale(x)
        return torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1)
        # NOTE The below allows using analytical KL divergence directly
        # cov = torch.diag_embed(scale ** 2)  # [B, D] B batches of D scales --> [B, D, D] B batches of diagonal DxD matrices
        # return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)


class BernoulliSample(Distribution):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        p = self.p.forward(x)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=p), 1)

    
class ContinuousBernoulliSample(Distribution):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        p = self.p(x)
        return p, (p,)
    
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
        raise NotImplementedError('Not yet adapted to GaussianSample layer using Softplus to return standard deviation')
        # Calculate precision of each distribution
        # (inverse variance)
        mu2 = self.mu(z)
        #log_var2 = F.softplus(self.log_var(z))
        log_var2 = self.log_var(z)
        precision1, precision2 = (1 / torch.exp(log_var1), 1 / torch.exp(log_var2))

        # Merge distributions into a single new
        # distribution
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
