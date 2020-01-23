import torch.nn as nn

import vari.models


def build_dense_vae(vae_class, **kwargs):
    if vae_class == vari.models.HierarchicalVariationalAutoencoder:
        x_dim, z_dim, h_dim = kwargs['x_dim'], kwargs['z_dim'], kwargs['h_dim']
        encoder_distribution = [getattr(vari.layers, distribution) for distribution in kwargs['encoder_distribution']]
        decoder_distribution = [getattr(vari.layers, distribution) for distribution in kwargs['decoder_distribution']]
        
        enc_dims = [x_dim, *z_dim]
        dec_dims = enc_dims[::-1]  # reverse
        h_dim_rev = h_dim[::-1]  # reverse
        
        vae_kwargs = dict(
            encoder=nn.ModuleList([
                vari.models.DenseSequentialCoder(
                    x_dim=enc_dims[i - 1],
                    z_dim=enc_dims[i],
                    h_dim=h_dim[i - 1],
                    distribution=encoder_distribution[i - 1],
                    activation=nn.Tanh)
                for i in range(1, len(enc_dims))
            ]),
            decoder=nn.ModuleList([
                vari.models.DenseSequentialCoder(
                    x_dim=dec_dims[i - 1],
                    z_dim=dec_dims[i],
                    h_dim=h_dim_rev[i - 1],
                    distribution=decoder_distribution[i - 1],
                    activation=nn.Tanh)
                for i in range(1, len(dec_dims))
            ])
        )
        return vari.models.HierarchicalVariationalAutoencoder(**vae_kwargs), vae_kwargs
    
    raise NotImplementedError(f'Not implemented for {vae_class}')
