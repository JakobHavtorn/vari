import torch.nn as nn

import vari.models


def build_dense_vae(x_dim, z_dim, h_dim, encoder_distribution, decoder_distribution, activation=nn.Tanh):
    encoder_distribution = [getattr(vari.layers, distribution) for distribution in encoder_distribution]
    decoder_distribution = [getattr(vari.layers, distribution) for distribution in decoder_distribution]
    
    assert len(z_dim) == len(h_dim)
    assert len(z_dim) == len(encoder_distribution)
    assert len(encoder_distribution) == len(decoder_distribution)
    
    enc_dims = [x_dim, *z_dim]
    dec_dims = enc_dims[::-1]  # reverse
    h_dim_rev = [h[::-1] for h in h_dim][::-1]  # reverse [[a, b], [c, d]] --> [[d, c], [b, a]]
    
    vae_kwargs = dict(
        encoder=nn.ModuleList([
            vari.models.DenseSequentialCoder(
                x_dim=enc_dims[i - 1],
                z_dim=enc_dims[i],
                h_dim=h_dim[i - 1],
                distribution=encoder_distribution[i - 1],
                activation=activation)
            for i in range(1, len(enc_dims))
        ]),
        decoder=nn.ModuleList([
            vari.models.DenseSequentialCoder(
                x_dim=dec_dims[i - 1],
                z_dim=dec_dims[i],
                h_dim=h_dim_rev[i - 1],
                distribution=decoder_distribution[i - 1],
                activation=activation)
            for i in range(1, len(dec_dims))
        ])
    )
    return vari.models.HierarchicalVariationalAutoencoder(**vae_kwargs), vae_kwargs
