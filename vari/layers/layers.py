import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    # def __repr__(self):
    #     return inspect.getsourcelines(self.lambd)[0][0].replace('\n', '')


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and sd.
    """
    def reparametrize(self, mu, sd):
        epsilon = torch.randn_like(mu, requires_grad=False, device=get_device())
        # # log_std = 0.5 * log_var
        # # std = exp(log_std)
        # std = log_var.mul(0.5).exp_()

        # # z = std * epsilon + mu
        z = mu.addcmul(sd, epsilon)
        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
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

    def forward(self, x):
        mu = self.mu(x)
        scale = self.scale(x)
        return self.reparametrize(mu, scale), (mu, scale)

    def log_likelihood(self, x, mu, sd):
        return log_gaussian(x, mu, sd)

    
class BernoulliSample(Stochastic):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()
        
    def reparametrize(self, p):
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=p)
        return bernoulli.sample(sample_shape=p.shape)

    def forward(self, x):
        p = self.activation(self.p.forward(x))
        return p, (p,)

    def log_likelihood(self, x, p):
        return log_bernoulli(x, p)
    
    
class ContinuousBernoulliSample(Stochastic):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.p = nn.ModuleList([
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        ])

    def reparametrize(self, p):
        u = torch.rand_like(p, requires_grad=False, device=get_device())
        # For p != 0.5
        (torch.log(u * (2 * p - 1) + 1 - p) - torch.log(1 - p)) / (torch.log(p) - torch.log(1 - p))
        # For p == 0.5
        u
        return z
        
    def forward(self, x):
        p = self.p(x)
        return p
    
    def log_likelihood(self, x, p):
        return log_continuous_bernoulli(x, p)
        


class GaussianMerge(GaussianSample):
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


class GumbelSoftmax(Stochastic):
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
