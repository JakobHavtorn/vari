import torch.nn as nn

from vari.models.vae import DenseSequentialCoder

from vari.layers import GaussianSample, BernoulliSample, ContinuousBernoulliSample


def get_default_model_config(vae_type, dataset):
    if dataset in ['MNISTBinarized', 'FashionMNISTBinarized', 'MNISTReal', 'FashionMNISTReal']:
        kwargs = get_default_model_config_mnist(vae_type)
    elif dataset in ['Moons', 'Spirals']:
        kwargs = get_default_model_config_synthetic_2d(vae_type)
    return kwargs


def get_default_model_config_mnist(vae_type):
    # LeakyReLU for MNIST models
    if vae_type == 'VariationalAutoencoder':
        x_dim, z_dim, h_dim = 784, 2, [512, 512, 256, 256]
        vae_kwargs = dict(
            encoder=DenseSequentialCoder(
                x_dim=x_dim,
                z_dim=z_dim,
                h_dim=h_dim,
                distribution=GaussianSample,
                activation=nn.LeakyReLU
            ),
            decoder=DenseSequentialCoder(
                x_dim=z_dim,
                z_dim=x_dim,
                h_dim=list(reversed(h_dim)),
                distribution=BernoulliSample,
                activation=nn.LeakyReLU
            )
        )
    elif vae_type == 'HierarchicalVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=784,
            z_dim=[5, 2],
            h_dim=[[512, 512], [256, 256]],
            encoder_distribution=[GaussianSample, GaussianSample],
            decoder_distribution=[GaussianSample, BernoulliSample],
            activation=nn.LeakyReLU
        )
    elif vae_type == 'AuxilliaryVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=784,
            z_dim=2,
            a_dim=2,
            h_dim=[512, 512, 256, 256],
            encoder_distribution=GaussianSample,
            decoder_distribution=BernoulliSample,
            activation=nn.LeakyReLU
        )
    elif vae_type == 'LadderVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=784,
            z_dim=[2, 2],
            h_dim=[512, 512],
            encoder_distribution=GaussianSample,
            decoder_distribution=BernoulliSample,
            activation=nn.LeakyReLU
        )
    return vae_kwargs


def get_default_model_config_synthetic_2d(vae_type):
    if vae_type == 'VariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=2,
            h_dim=[64, 64, 32, 32],
            encoder_distribution=GaussianSample,
            decoder_distribution=GaussianSample,
        )
    elif vae_type == 'HierarchicalVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=[2, 2],
            h_dim=[[64, 64], [32, 32]],
            encoder_distribution=[GaussianSample, GaussianSample],
            decoder_distribution=[GaussianSample, GaussianSample],
        )
    elif vae_type == 'AuxilliaryVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=2,
            a_dim=2,
            h_dim=[64, 64, 32, 32],
            encoder_distribution=GaussianSample,
            decoder_distribution=GaussianSample,
        )
    elif vae_type == 'LadderVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=[2, 2],
            h_dim=[64, 64, 32, 32],
            encoder_distribution=GaussianSample,
            decoder_distribution=GaussianSample,
        )
    return vae_kwargs
