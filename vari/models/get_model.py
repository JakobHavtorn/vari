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
        x_dim, z_dim, h_dim = 784, [5, 2], [[512, 512], [256, 256]]
        encoder_distribution = [GaussianSample, GaussianSample]
        decoder_distribution = [GaussianSample, BernoulliSample]
        activation = nn.LeakyReLU

        enc_dims = [x_dim, *z_dim]
        dec_dims = enc_dims[::-1]  # reverse
        h_dim_rev = h_dim[::-1]  # reverse
        
        vae_kwargs = dict(
            encoder=nn.ModuleList([DenseSequentialCoder(
                x_dim=enc_dims[i - 1],
                z_dim=enc_dims[i],
                h_dim=h_dim[i - 1],
                distribution=encoder_distribution[i - 1],
                activation=activation) for i in range(1, len(enc_dims))]),
            decoder=nn.ModuleList([DenseSequentialCoder(
                x_dim=dec_dims[i - 1],
                z_dim=dec_dims[i],
                h_dim=h_dim_rev[i - 1],
                distribution=decoder_distribution[i - 1],
                activation=activation) for i in range(1, len(dec_dims))])
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
        x_dim, z_dim, h_dim = 2, 2, [64, 64, 32, 32]
        vae_kwargs = dict(
            encoder=DenseSequentialCoder(
                x_dim=x_dim,
                z_dim=z_dim,
                h_dim=h_dim,
                distribution=GaussianSample,
                activation=nn.Tanh
            ),
            decoder=DenseSequentialCoder(
                x_dim=z_dim,
                z_dim=x_dim,
                h_dim=list(reversed(h_dim)),
                distribution=GaussianSample,
                activation=nn.Tanh
            )
        )
    elif vae_type == 'HierarchicalVariationalAutoencoder':
        x_dim, z_dim, h_dim = 2, [2, 2], [[64, 64], [32, 32]]
        encoder_distribution = [GaussianSample, GaussianSample]
        decoder_distribution = [GaussianSample, GaussianSample]
        activation = nn.Tanh

        enc_dims = [x_dim, *z_dim]
        dec_dims = enc_dims[::-1]  # reverse
        h_dim_rev = h_dim[::-1]  # reverse
        
        vae_kwargs = dict(
            encoder=nn.ModuleList([DenseSequentialCoder(
                x_dim=enc_dims[i - 1],
                z_dim=enc_dims[i],
                h_dim=h_dim[i - 1],
                distribution=encoder_distribution[i - 1],
                activation=activation) for i in range(1, len(enc_dims))]),
            decoder=nn.ModuleList([DenseSequentialCoder(
                x_dim=dec_dims[i - 1],
                z_dim=dec_dims[i],
                h_dim=h_dim_rev[i - 1],
                distribution=decoder_distribution[i - 1],
                activation=activation) for i in range(1, len(dec_dims))])
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
