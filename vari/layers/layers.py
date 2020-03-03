import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

from torch.autograd import Variable

from vari.distributions import ContinuousBernoulli
from vari.layers.convenience import AddConstant, Clamp, Identity
from vari.utilities import get_device


class LearnableDistribution(nn.Module):
    def initialize(self):
        raise NotImplementedError
    
    @property
    def default_prior(self):
        raise NotImplementedError


class GaussianLayer(LearnableDistribution):
    r"""Layer that parameterizes a diagonal covariance Gaussian distribution by its mean and standard deviation vectors.
    
    The distribution is continuous and support [-inf, inf].
    
    The layer outputs a torch.distributions.Independent wrapped torch.distribution.Normal parameterized by the mean and
    standard deviation.
    
    The mean is parameterized by a linear transformation of the input without nonlinearity and has range [-inf, inf].
    The standard deviation is parameterized by a linear transformation of the input followed by a softplus activation
    scaling it to [0, inf] and then an AddConstant and/or Clamping to fix it in the range [min_sd, max_sd].
    
    TODO Rename scale to sd
    """
    def __init__(self, in_features, out_features, min_sd=1e-8, max_sd=10, prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.min_sd = min_sd
        self.max_sd = max_sd
        self.prior = self.default_prior if prior is None else prior

        self.mu = nn.Linear(in_features, out_features)
        self.scale = self.build_sd_parameter(min_sd, max_sd)
        self.initialize()

    def build_sd_parameter(self, min_val, max_val):
        sd = [nn.Linear(self.in_features, self.out_features), nn.Softplus()]
        if min_val is not None and min_val != 0:
            sd.append(AddConstant(min_val))
        if max_val is not None:
            sd.append(Clamp(max=max_val))
        return nn.Sequential(*sd)

    @property
    def default_prior(self):
        # Standard Gaussian prior N(0,1)
        return torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.zeros(self.out_features).to(get_device()),
                scale=torch.ones(self.out_features).to(get_device())
            ),
            1
        )

    @property
    def mixture_prior(self):
        # Mixture with some number of components
        raise NotImplementedError

    def initialize(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.scale[0].weight, gain=gain)  # For scale gain is ReLU's since it's close to SoftPlus
        nn.init.xavier_normal_(self.mu.weight, gain=1.)  # For mean there is no nonlinearity so take gain to be 1

    def forward(self, x):
        mu = self.mu(x)
        scale = self.scale(x)
        return torch.distributions.Independent(torch.distributions.Normal(mu, scale), x.ndim - 1)
        
    def extra_repr(self):
        s = 'kwargs={\n'
        for k in ['min_sd', 'max_sd']:
            s += f'  {k}={getattr(self, k)},\n'
        s = s[:-2] + '\n}'
        return s
    
    
class BetaLayer(LearnableDistribution):
    r"""Layer that parameterizes a number of independent Beta distributions by the alpha and beta vectors.
    
    The distribution is continuous and has support [0, 1].
    
    The layer outputs a torch.distributions.Independent wrapped torch.distribution.Beta.
    
    Both the alpha and beta parameter is parameterized by a linear tranformation followed by a Softplus activation and
    potentially clamps the minimum and/or maximum value.
    
    Note that the Beta distribution is bimodal if α, β < 1. 
    
    The mode is
     - \frac{\alpha-1}{\alpha+\beta-2} for α, β >1
     - any value in (0,1) for α, β = 1
     - 0 for α ≤ 1, β > 1
     - 1 for α > 1, β ≤ 1
     - {0, 1} (bimodal) for α, β < 1
    """
    def __init__(self, in_features, out_features, min_alpha=1e-3, max_alpha=None, min_beta=1e-3, max_beta=None,
                 prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.min_alpha = min_alpha
        self.min_beta = min_beta
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.prior = self.default_prior if prior is None else prior

        self.alpha = self.build_concentration_parameter(min_alpha, max_alpha)
        self.beta = self.build_concentration_parameter(min_beta, max_beta)
        # self.mu = nn.Linear(in_features, out_features)
        # var = [nn.Linear(in_features, out_features), nn.Softplus()]
        # if min_beta is not None or max_beta is not None:
        #     var.append(Clamp(min=min_beta, max=max_beta))
        # self.var = nn.Sequential(*var)
        self.initialize()

    def build_concentration_parameter(self, min_val, max_val):
        concentration = [nn.Linear(self.in_features, self.out_features), nn.Softplus()]
        if min_val is not None and min_val != 0:
            concentration.append(AddConstant(min_val))
        if max_val is not None:
            concentration.append(Clamp(max=max_val))
        return nn.Sequential(*concentration)

    @property
    def default_prior(self):
        # Uniform distribution by alpha = beta = 1
        return torch.distributions.Independent(
            torch.distributions.Beta(
                concentration0=torch.ones(self.out_features).to(get_device()),
                concentration1=torch.ones(self.out_features).to(get_device())
            ),
            1
        )

    def initialize(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.alpha[0].weight, gain=gain)
        nn.init.xavier_normal_(self.beta[0].weight, gain=gain)
        # nn.init.xavier_normal_(self.mu.weight, gain=1)
        # nn.init.xavier_normal_(self.var[0].weight, gain=gain)

    def forward(self, x):
        alpha = self.alpha(x)
        beta = self.beta(x)
        # mu = self.mu(x)
        # var = self.var(x)
        # tmp = mu * (1 - mu)
        # var[var >= tmp] = tmp[var >= tmp] - 1e3
        # nu = tmp / var - 1
        # alpha = mu * nu
        # beta = (1 - mu) * nu
        return torch.distributions.Independent(torch.distributions.Beta(alpha, beta), x.ndim - 1)

    def extra_repr(self):
        s = 'kwargs={\n'
        for k in ['min_alpha', 'max_alpha', 'min_beta', 'max_beta']:
            s += f'  {k}={getattr(self, k)},\n'
        s = s[:-2] + '\n}'
        return s


class GaussianFixedVarianceLayer(LearnableDistribution):
    """Layer that parameterizes a Gaussian distribution by its mean and a fixed standard deviation.
    
    The layer outputs a torch.distributions.Independent wrapped torch.distribution.Normal parameterized by the mean and
    standard deviation.
    
    The mean is parameterized by a linear transformation of the input without nonlinearity and has range [-inf, inf].
    The standard deviation is fixed at a constant value of `std` and is not learnable.
    As such, the layer can only model homoscedastic variance.
    
    The log-likelihod of the distributions resulting from this layer correspond to the MSE loss function. 
    
    If the std is set to sqrt(0.5) then the MSE loss is not scaled but is offset by -0.5*log(2π * sqrt(0.5)) = -0.746.
    If the std is set to 1, then the MSE loss is scaled by 0.5 and is offset by -0.5*log(2π) = 0.919.
    From a modelling perspective, setting the standard deviation to the standard deviation of the input space data
    distribution makes the most sense, but this may not be known or may be variable depending on the input value
    (heteroscedastic variance) in which case learnable variance might be more sensible.
    """
    def __init__(self, in_features, out_features, std=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std = std
        
        self.prior = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.zeros(self.out_features).to(get_device()),
                scale=std * torch.ones(out_features).to(get_device())
                ),
            reinterpreted_batch_ndims=1
        )

        self.mu = nn.Linear(in_features, out_features)
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_normal_(self.mu.weight, gain=1.)  # For mean there is no nonlinearity so take gain to be 1

    def forward(self, x):
        mu = self.mu(x)
        return torch.distributions.Independent(torch.distributions.Normal(mu, self.scale), x.ndim - 1)
    
    def extra_repr(self):
        s = 'kwargs={\n'
        for k in ['std']:
            s += f'  {k}={getattr(self, k)},\n'
        s = s[:-2] + '\n}'
        return s


class BernoulliLayer(LearnableDistribution):
    """Layer that parameterizes a Bernoulli distribution.
    
    Args:
        in_features (int): Number if input features. If `None`, the inputs to `forward` will be wrapped directly.
        out_features (int): Number if output features.
    """
    def __init__(self, in_features=None, out_features=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if in_features is not None:
            self.p = nn.Linear(in_features, out_features)
        else:
            self.p = Identity()
        self.initialize()
    
    def initialize(self):
        if self.in_features is not None:
            gain = nn.init.calculate_gain('sigmoid')  # Bernoulli distribution converts logits to probs with sigmoid
            nn.init.xavier_normal_(self.p.weight, gain=gain)

    def forward(self, x):
        p = self.p(x)
        return torch.distributions.Independent(torch.distributions.Bernoulli(logits=p), x.ndim - 1)


class ContinuousBernoulliLayer(BernoulliLayer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features)
        
    def forward(self, x):
        p = self.p(x)
        return torch.distributions.Independent(ContinuousBernoulli(logits=p), x.ndim - 1)
    

class GaussianMerge(LearnableDistribution):
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


class GumbelSoftmax(LearnableDistribution):
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
