import torch

from vari.inference import log_gaussian, log_standard_gaussian


def kld_gaussian_gaussian(z, q_param, p_param=None):
    """
    Computes the KL-divergence of
    some element z.
    KL(q||p) = ∫ q(z) log [ q(z) / p(z) ]
             = E[log q(z) - log p(z)]
    :param z: sample from q-distribuion
    :param q_param: (mu, sd) of the q-distribution
    :param p_param: (mu, sd) of the p-distribution
    :return: KL(q||p)
    """
    (q_mu, q_sd) = q_param
    qz = log_gaussian(z, q_mu, q_sd)
    if p_param is None:
        pz = log_standard_gaussian(z)
    else:
        p_mu, p_sd = p_param
        pz = log_gaussian(z, p_mu, p_sd)
    kl = qz - pz
    return kl


def kld_gaussian_gaussian_analytical(q_param, p_param=None):
    """
    Computes the KL-divergence of
    some element z.
    KL(q||p) = ∫ q(z) log [ q(z) / p(z) ]
             = E[log q(z) - log p(z)]
    :param z: sample from q-distribuion
    :param q_param: (mu, sd) of the q-distribution
    :param p_param: (mu, sd) of the p-distribution
    :return: KL(q||p)
    """
    raise NotImplementedError()
    (q_mu, q_sd) = q_param
    if p_param is None:
        p_mu = torch.zeros_like(q_mu)
        p_sd = torch.ones_like(q_sd)
    else:
        p_mu, p_sd = p_param

    return kl
