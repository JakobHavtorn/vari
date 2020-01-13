import math
import torch
import torch.nn.functional as F


# def asinh(x):
#     return torch.log(x+(x**2+1)**0.5)

# def acosh(x):
#     return torch.log(x+(x**2-1)**0.5)

def atanh(x):
    """Computes hyperbolic tangent for x in (-1, 1)
    
    NOTE May not be numerically stable
    """
    return 0.5 * torch.log(1 + x) - torch.log(1 - x)


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, sd):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and sd evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param sd: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - sd.log() - (x - mu)**2 / (2 * sd**2)
    # log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_bernoulli(x, p, eps=1e-8):
    """
    Returns the log pdf of a Bernoulli distribution parametrised
    by p evaluated at x.

    :param x: point to evaluate
    :param p: mean of distribution
    :return: log B(x|p)
    """
    log_pdf = (p + eps).log() * x + (1-p+eps).log() * (1-x)
    # bernoulli = torch.distributions.bernoulli.Bernoulli(probs=p)
    # bernoulli.log_prob(x)
    return torch.sum(log_pdf, dim=-1)

def log_continuous_bernoulli(x, p):
    log_pdf = log_bernoulli(x, p)
    log_C = torch.log(2 * atanh(1 - 2*p)) - torch.log(1 - 2*p)
    return torch.sum(log_pdf + log_C, dim=-1)


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy
