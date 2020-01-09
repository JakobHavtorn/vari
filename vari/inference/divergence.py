import torch

from vari.inference import log_gaussian, log_standard_gaussian


def kld_gaussian_gaussian(z, q_param, p_param=None):
    """
    Computes the KL-divergence of
    some element z.
    KL(q||p) = ∫ q(z) log [ q(z) / p(z) ]
             = E[log q(z) - log p(z)]
    :param z: sample from q-distribuion
    :param q_param: (mu, log_var) of the q-distribution
    :param p_param: (mu, log_var) of the p-distribution
    :return: KL(q||p)
    """
    (q_mu, q_log_var) = q_param
    qz = log_gaussian(z, q_mu, q_log_var)
    if p_param is None:
        pz = log_standard_gaussian(z)
    else:
        p_mu, p_log_var = p_param
        pz = log_gaussian(z, p_mu, p_log_var)
    kl = qz - pz
    return kl


def kld_gaussian_gaussian_analytical(q_param, p_param=None):
    """
    Computes the KL-divergence of
    some element z.
    KL(q||p) = ∫ q(z) log [ q(z) / p(z) ]
             = E[log q(z) - log p(z)]
    :param z: sample from q-distribuion
    :param q_param: (mu, log_var) of the q-distribution
    :param p_param: (mu, log_var) of the p-distribution
    :return: KL(q||p)
    """
    raise NotImplementedError()
    (q_mu, q_log_var) = q_param
    if p_param is None:
        p_mu = torch.zeros_like(q_mu)
        p_log_var = torch.ones_like(q_log_var)
    else:
        p_mu, p_log_var = p_param

    # kl = torch.sum(p_log_var - q_log_var + (torch.exp(q_log_var) + (q_mu - p_mu)**2) / (2 * torch.exp(p_log_var)) - 0.5, dim=-1)

    return kl
