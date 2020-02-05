import torch.nn as nn

import vari.models


def build_dense_vae(x_dim, z_dim, h_dim, encoder_distribution, decoder_distribution, encoder_distribution_kwargs=None,
                    decoder_distribution_kwargs=None, activation=nn.Tanh):
    encoder_distribution = [getattr(vari.layers, distribution) for distribution in encoder_distribution]
    if encoder_distribution_kwargs is None:
        encoder_distribution_kwargs = [{}] * len(encoder_distribution)

    decoder_distribution = [getattr(vari.layers, distribution) for distribution in decoder_distribution]
    if decoder_distribution_kwargs is None:
        decoder_distribution_kwargs = [{}] * len(decoder_distribution)
    
    assert len(z_dim) == len(h_dim)
    assert len(z_dim) == len(encoder_distribution)
    assert len(encoder_distribution) == len(decoder_distribution)
    
    enc_dims = [x_dim, *z_dim]
    dec_dims = enc_dims[::-1]  # reverse
    h_dim_rev = [h[::-1] for h in h_dim][::-1]  # reverse [[a, b], [c, d]] --> [[d, c], [b, a]]

    # TODO Write this as an explicit for loop
    vae_kwargs = dict(
        encoder=nn.ModuleList([
            vari.models.DenseSequentialCoder(
                x_dim=enc_dims[i - 1],
                h_dim=h_dim[i - 1],
                distribution=encoder_distribution[i - 1](
                    in_features=h_dim[i - 1][-1],
                    out_features=enc_dims[i],
                    **encoder_distribution_kwargs[i - 1]
                ),
                activation=activation
            )
            for i in range(1, len(enc_dims))
        ]),
        decoder=nn.ModuleList([
            vari.models.DenseSequentialCoder(
                x_dim=dec_dims[i - 1],
                h_dim=h_dim_rev[i - 1],
                distribution=decoder_distribution[i - 1](
                    in_features=h_dim_rev[i - 1][-1],
                    out_features=dec_dims[i],
                    **decoder_distribution_kwargs[i - 1]
                ),
                activation=activation
            )
            for i in range(1, len(dec_dims))
        ])
    )
    return vari.models.HierarchicalVariationalAutoencoder(**vae_kwargs), vae_kwargs
