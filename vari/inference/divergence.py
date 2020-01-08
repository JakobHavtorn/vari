import torch

from vari.inference import log_gaussian, log_standard_gaussian


def kld_gaussian_gaussian(z, q_param, p_param=(0, 1)):
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
    (mu, log_var) = q_param

    qz = log_gaussian(z, mu, log_var)

    if p_param == (0, 1):
        pz = log_standard_gaussian(z)
    else:
        (mu, log_var) = p_param
        pz = log_gaussian(z, mu, log_var)

    kl = qz - pz

    return kl


def kld_gaussian_gaussian_analytical(z, q_param, p_param=(0, 1)):
    # TODO Implement deterministic KL divergence between univariate gaussians
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
    (p_mu, p_log_var) = p_param
    
    kl = p_log_var - q_log_var + (torch.exp(q_log_var) + (q_mu - p_mu)**2) / (2 * torch.exp(p_log_var)) - 0.5

    return kl
