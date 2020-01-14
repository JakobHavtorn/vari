import vari.models.vae

from vari.layers import GaussianSample, BernoulliSample, ContinuousBernoulliSample


def get_default_model_config(vae_type, dataset):
    if dataset == 'MNIST':
        kwargs = get_default_model_config_mnist(vae_type)
    elif dataset in ['Moons', 'Spirals']:
        kwargs = get_default_model_config_synthetic_2d(vae_type)
    return kwargs


def get_default_model_config_mnist(vae_type):
    if vae_type == 'VariationalAutoencoder':
        vae_kwargs = dict(
            z_dim=2,
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=BernoulliSample,
        )
    elif vae_type == 'HierarchicalVariationalAutoencoder':
        vae_kwargs = dict(
            z_dim=[2, 2],
            h_dim=[64, 64],
            encoder_sample_layer=[GaussianSample, GaussianSample],
            decoder_sample_layer=[GaussianSample, BernoulliSample],
        )
    elif vae_type == 'AuxilliaryVariationalAutoencoder':
        vae_kwargs = dict(
            z_dim=2,
            a_dim=2,
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=BernoulliSample,
        )
    elif vae_type == 'LadderVariationalAutoencoder':
        vae_kwargs = dict(
            z_dim=[2, 2],
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=BernoulliSample,
        )
    return vae_kwargs


def get_default_model_config_synthetic_2d(vae_type):
    if vae_type == 'VariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=2,
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=GaussianSample,
        )
    elif vae_type == 'HierarchicalVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=[2, 2],
            h_dim=[64, 64],
            encoder_sample_layer=[GaussianSample, GaussianSample],
            decoder_sample_layer=[GaussianSample, GaussianSample],
        )
    elif vae_type == 'AuxilliaryVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=2,
            a_dim=2,
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=GaussianSample,
        )
    elif vae_type == 'LadderVariationalAutoencoder':
        vae_kwargs = dict(
            x_dim=2,
            z_dim=[2, 2],
            h_dim=[64, 64],
            encoder_sample_layer=GaussianSample,
            decoder_sample_layer=GaussianSample,
        )
    return vae_kwargs