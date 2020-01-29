from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vari.layers import Identity, GaussianLayer, BernoulliLayer, GaussianMerge, GumbelSoftmax
from vari.inference import log_gaussian, log_standard_gaussian
from vari.inference.divergence import kld_gaussian_gaussian, kld_gaussian_gaussian_analytical
from vari.utilities import get_device, log_sum_exp


class MultiLayeredPerceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super().__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)
        return x


class DenseSequentialCoder(nn.Module):
    """
    Coder network that can parameterize a distribution, variational or not.
    
    If used as "inference network", attempts to infer the probability distribution p_true(z|x) from the data by fitting a
    variational distribution q(z|x). 
    
    If used as generative network, attempts to infer the probability distribution p_true(x|z) from the data by fitting a 
    distribution p(x|z).
    
    Returns the distribution (with batch dimension) that encodes a batch of examples (x or z).

    Arguments:
        x_dim (int): Dimensionality of the inputs to be expected e.g. 784
        z_dim (int); Dimensionality of the latent space (regularized to conform to the prior of the `distribution`)
        h_dim (list of int): Dimensionality of the intermediate hidden affine nonlinear transformations.
        activation (nn.Activation): The activation function to apply between affine layers.
        distribution (vari.layers): The model that parameterizes the distribution of the latent space.
    """
    def __init__(self, x_dim, z_dim, h_dim, activation=nn.Tanh, distribution=GaussianLayer):
        super().__init__()
        assert isinstance(x_dim, int) and isinstance(z_dim, int) and isinstance(h_dim, list)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self._activation = activation
        self._distribution = distribution

        modules = []
        dims = [x_dim, *self.h_dim]
        for i in range(1, len(dims)):
            modules.extend([
                nn.Linear(dims[i-1], dims[i]),
                activation(),
            ])
        self.coder = nn.Sequential(*modules)
        self.distribution = distribution(self.h_dim[-1], z_dim)
        self.initialize()

    def initialize(self):
        activation = self._activation().__class__.__name__.lower() if not isinstance(self._activation(), nn.LeakyReLU) else 'leaky_relu'
        gain = nn.init.calculate_gain(activation, param=None)
        i = 0
        for m in self.coder.modules():
            if isinstance(m, nn.Linear):
                print(i, m)
                i += 1
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """Forward pass an example through the model
        
        Args:
            x (tensor): Tensor of shape [B, D] where B is a number of batch dimensions and D is len(self.x_dim) features
        
        Returns:
            tensor: Tensor of shape [B, product(self.x_dim)] where the feature dimensions have been flattened.
        """
        h = self.coder(x)
        return self.distribution(h)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_divergences = OrderedDict([('z', 0)])

    @property
    def kl_divergence(self):
        return sum(self.kl_divergences.values())

    def elbo(self, x, importance_samples=1, beta=1, reduce_importance_samples=True, analytical_kl=False):
        """Computes a complete forward pass and then computes the log-likelihood log p(x|z) and the ELBO log p(x)
        and returns the ELBO, log-likelihood and the total KL divergence.

        Args:
            x (tensor): Inputs of shape [N, D] where N is batch and D is input dimension (potentially more than one)
            importance_samples (int, optional): Number of importance samples to use. Defaults to None.
            beta (int, optional): Value of ß for the ß-VAE or deterministic warmup parameter. Defaults to 1.
            analytical_kl (bool, optional): Whether to compute KL divergence between q(z|x) and p(z) analytically.
            reduce_importance_samples (bool, optional): Whether to return elbo, likelihood and KL reduced over samples.

        Returns:
            [type]: [description]
        """
        px = self.forward(x, importance_samples=importance_samples, analytical_kl=analytical_kl)
        likelihood = px.log_prob(x.view(-1, *px.event_shape))
        elbo = likelihood - beta * self.kl_divergence

        if reduce_importance_samples:
            return self.reduce_importance_samples(elbo, likelihood, self.kl_divergences)
        return elbo, likelihood, self.kl_divergence

    def reduce_importance_samples(self, elbo, likelihood, kl_divergences):
        self.kl_divergences = OrderedDict([(k, kl.mean(axis=0)) for k, kl in kl_divergences.items()])
        # return log_sum_exp(elbo, axis=0, sum_op=torch.mean).flatten(), likelihood.mean(axis=0), self.kl_divergence
        # self.kl_divergences = OrderedDict([(k, log_sum_exp(kl, axis=0, sum_op=torch.mean).flatten()) for k, kl in self.kl_divergences.items()])
        return log_sum_exp(elbo, axis=0, sum_op=torch.mean).flatten(), likelihood.mean(axis=0), self.kl_divergence

    def encode(self, x, importance_samples=1):
        qz = self.encoder(x)
        z = qz.rsample(torch.Size([importance_samples]))
        return dict(z=dict(sample=z, distribution=qz))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, y=None, importance_samples=1, analytical_kl=False):
        """
        Runs a data point through the model in order to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        assert not analytical_kl or (analytical_kl and importance_samples == 1), \
            'KL is not analytically computable with importance sampling'

        latent = self.encode(x, importance_samples=importance_samples)  # Latent inference q(z|x)
        z, qz = latent['z']['sample'], latent['z']['distribution']
        px = self.decode(z)  # Generative p(x|z)
        if analytical_kl:
            self.kl_divergences['z'] = torch.distributions.kl_divergence(qz, self.encoder.distribution.get_prior())[None, ...]
        else:
            self.kl_divergences['z'] = qz.log_prob(z) - self.encoder.distribution.get_prior().log_prob(z)
        return px

    def generate(self, n_samples=None, z=None, seed=None):
        """
        Generate samples from the generative model by either sampling `n_samples` from the prior p(z) or by decoding
        the given latent representation `z`. In both cases, setting `seed` can make decoding reproducible.
        """
        assert (n_samples is not None) != (z is not None), 'Specify either n_samples or z.'
        if seed is not None:
            torch.manual_seed(seed)
        if z is not None:
            return self.decode(z)
        z = self.encoder.distribution.get_prior().sample(torch.Size([n_samples]))
        return self.decode(z)


class HierarchicalVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_layers = len(encoder)
        self.kl_divergences = OrderedDict([(f'z{i+1}', 0) for i in range(0, self.n_layers)])

    @property
    def kl_divergence(self):
        return sum(self.kl_divergences.values())

    def elbo(self, x, importance_samples=1, beta=1, reduce_importance_samples=True, copy_latents=None):
        """Computes a complete forward pass and then computes the log-likelihood log p(x|z) and the ELBO log p(x)
        and returns the ELBO, log-likelihood and the total KL divergence.

        Args:
            x (tensor): Inputs of shape [N, *, D] where N is batch and D is input dimension (potentially more than one)
            importance_samples ([type], optional): [description]. Defaults to None.
            beta (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        px = self.forward(x, importance_samples=importance_samples, copy_latents=copy_latents)
        likelihood = px.log_prob(x.view(-1, *px.event_shape))
        elbo = likelihood - beta * self.kl_divergence

        if reduce_importance_samples:
            return self.reduce_importance_samples(elbo, likelihood, self.kl_divergences)
        return elbo, likelihood, self.kl_divergence

    def reduce_importance_samples(self, elbo, likelihood, kl_divergences):
        self.kl_divergences = OrderedDict([(k, kl.mean(axis=0)) for k, kl in kl_divergences.items()])
        return log_sum_exp(elbo, axis=0, sum_op=torch.mean).flatten(), likelihood.mean(axis=0), self.kl_divergence

    def encode(self, x, importance_samples=1):
        """Return list of latents with an element being a tuple of samples and a tuple of the parameters of the q(z|x)
        """
        latents = OrderedDict()
        z = x
        for i, encoder in enumerate(self.encoder, start=1):
            qz = encoder(z)
            z = qz.rsample(torch.Size([importance_samples])) if i==1 else qz.rsample()
            latents[f'z{i}'] = (z, qz)
        return latents

    def decode(self, latents, copy_latents=None):
        """Decode a list of latent representations
        
        Args:
            latents (list of tuple): List of samples and distribution parameters [(z1, z1_distribution), ...]
            copy_latents (dict, optional): If not None must be a dict that is True for each latent to copy from
                                           the encoder and False when wanting to use the generative sample.
                                           If the top most latent is not copied, we sample from the prior in the same
                                           dimensions as the top most encoded latent.
                                           Defaults to None in which case all latents are copied from the encoder.

        Returns:
            tuple: Tuple of samples and distribution parameters for input space [(x, (px_parameters)), ...]
        """
        assert copy_latents is None or len(copy_latents) == len(latents), 'Specify for each latent whether to copy.'
        self.kl_divergences = OrderedDict([(f'z{i+1}', 0) for i in range(0, self.n_layers)])  # Reset

        for z_index in range(self.n_layers, 0, -1):  # [self.n_layers, self.n_layers - 1, ..., 1]
            z_key = f'z{z_index}'
            qz_samples, qz = latents[z_key]
            if z_index == self.n_layers:  # At top we use prior for KL
                self.kl_divergences[z_key] = qz.log_prob(qz_samples) - \
                                                        self.encoder[-1].distribution.get_prior().log_prob(qz_samples)
            else:
                self.kl_divergences[z_key] = qz.log_prob(qz_samples) - pz.log_prob(qz_samples)
            if copy_latents is None or copy_latents[z_key]:  # Copy latents from layer below for decoding (and KL)
                pz = self.decoder[-z_index](qz_samples)  # p(z_{i-1}|z_{i}) where z_{i} ~  q(z_{i}|z_{i-1}) or q(z_1|x)
            else:
                if z_index == self.n_layers:
                    pz = self.encoder[-1].distribution.get_prior()
                    pz_samples = pz.rsample(qz.batch_shape)
                else:
                    pz_samples = pz.rsample()
                pz = self.decoder[-z_index](pz_samples)  # p(z_{i-1}|z_{i}) where z_{i} ~  p(z_{i}|z_{i+1})

        return pz  # Final pz is actually p(x|z)

    def forward(self, x, importance_samples=1, copy_latents=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        qz = self.encode(x, importance_samples=importance_samples)  # Latent inference q(z|x)
        px = self.decode(qz, copy_latents=copy_latents)  # Generative p(x|z)
        return px

    def generate(self, n_samples=None, z=None, seed=None):
        """
        Generate samples from the generative model by either sampling `n_samples` from the prior p(z) or by decoding
        the given latent representation `z`. In both cases, setting `seed` can make decoding reproducible.
        """
        def decode(z):
            for decoder in self.decoder:
                pz = decoder(z)
                z = pz.sample()
            return pz
        if seed is not None:
            torch.manual_seed(seed)
        if z is not None:
            return decode(z)
        z = self.encoder.distribution.get_prior().sample(torch.Size([n_samples]))
        return decode(z)


class AuxilliaryVariationalAutoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, a_dim, encoder_distribution=GaussianLayer, decoder_distribution=GaussianLayer, activation=nn.Tanh):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        super(AuxilliaryVariationalAutoencoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim

        self.aux_encoder = DenseSequentialCoder(x_dim=x_dim, z_dim=a_dim, h_dim=h_dim, activation=activation)
        self.aux_decoder = DenseSequentialCoder(x_dim=x_dim + z_dim, z_dim=a_dim, h_dim=list(reversed(h_dim)), activation=activation)

        self.encoder = DenseSequentialCoder(x_dim=a_dim + x_dim, z_dim=z_dim, h_dim=h_dim, distribution=encoder_distribution, activation=activation)
        self.decoder = DenseSequentialCoder(x_dim=z_dim, z_dim=x_dim, h_dim=list(reversed(h_dim)), distribution=decoder_distribution, activation=activation)
        
        self.kl_divergence = 0
        self.kl_divergences = [0] * 2
        self.initialize()

    def initialize(self):
        gain = nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        q_a, (q_a_mu, q_a_sd) = self.aux_encoder(x)
        q_z, (q_z_mu, q_z_sd) = self.encoder(torch.cat([x, q_a], dim=1))
        return (q_z, (q_z_mu, q_z_sd)), (q_a, (q_a_mu, q_a_sd))

    def decode(self, z):
        p_x, px_args = self.decoder(z)
        # p_a, pa_args = self.aux_decoder(torch.cat([p_x, z], dim=1))
        return (p_x, px_args)

    def forward(self, x):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, (q_a_mu, q_a_sd) = self.aux_encoder(x)

        # Latent inference q(z|a,x)
        q_z, (q_z_mu, q_z_sd) = self.encoder(torch.cat([x, q_a], dim=1))

        # Generative p(x|z)
        p_x, px_args = self.decoder(q_z)

        # Generative p(a|z,x)
        p_a, (p_a_mu, p_a_sd) = self.aux_decoder(torch.cat([p_x, q_z], dim=1))

        a_kl = kld_gaussian_gaussian(q_a, (q_a_mu, q_a_sd), (p_a_mu, p_a_sd))
        z_kl = kld_gaussian_gaussian(q_z, (q_z_mu, q_z_sd))
        a_kl.clamp_(min=0.1)  # Free bits
        self.kl_divergences[0], self.kl_divergences[1] = z_kl, a_kl
        self.kl_divergence = a_kl + z_kl

        return p_x, px_args
    
    def sample(self, z):
        return self.decode(z)
    
    def log_likelihood(self, x, *px_args):
        return self.decoder.sample.log_likelihood(x, *px_args)


class LadderEncoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.
        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianLayer(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.tanh(self.batchnorm(x))
        return x, self.sample(x)


class VariationalInferenceModel(nn.Module):
    """Model that encodes the generative PGM: z_N -> z_N-1 -> ... -> z_1 --> x without amortized inference of z_:.
    """
    def __init__(self):
        raise NotImplementedError()


class LadderDecoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.
        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()

        [self.z_dim, h_dim, x_dim] = dims

        self.linear1 = nn.Linear(x_dim, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, self.z_dim)

        self.linear2 = nn.Linear(x_dim, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianLayer(h_dim, self.z_dim)

    def forward(self, x, l_mu=None, l_sd=None):
        if l_mu is not None:
            # Sample from this encoder layer and merge
            z = self.linear1(x)
            z = F.tanh(self.batchnorm1(z))
            q_z, q_mu, q_sd = self.merge(z, l_mu, l_sd)

        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.tanh(self.batchnorm2(z))
        z, p_mu, p_sd = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_sd), (p_mu, p_sd))


class LadderVariationalAutoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.
        :param dims: x, z and hidden dimensions of the networks
        """
        super().__init__()

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = DenseSequentialCoder(x_dim=z_dim[0], z_dim=x_dim, h_dim=h_dim)
        self.initialize()

    def initialize(self):
        gain = nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        latents = []
        for encoder in self.encoder:
            x, (z, z_mu, z_sd) = encoder(x)
            latents.append((z, z_mu, z_sd))
        return latents

    def decode(self, latents):
        self.kl_divergence = 0
        for i, decoder in enumerate([None, *self.decoder]):
            _, l_mu, l_sd = latents[i]
            if i == 0:
                # If at top, encoder == decoder, use prior for KL.
                z = latents[i][0]
                self.kl_divergence += kld_gaussian_gaussian(z, (l_mu, l_sd))
            else:
                # Perform downward merge of information.
                z, kl = decoder(z, l_mu, l_sd)
                self.kl_divergence += kld_gaussian_gaussian(*kl)

        p_x, p_x_mu, p_x_sd = self.reconstruction(z)
        return p_x, p_x_mu, p_x_sd

    def forward(self, x):
        # Gather latent representation from encoders along with final z.
        latents = self.encode(x)
        latents = list(reversed(latents))
        p_x, p_x_mu, p_x_sd = self.decode(latents)
        return p_x, p_x_mu, p_x_sd

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)
    
    def log_likelihood(self, x, px_args):
        return self.decoder.sample.log_likelihood(x, px_args)


# NOTE First go at implementing variable number of hidden layers between stochastic variables
# class LadderEncoder(nn.Module):
#     """
#     Inference network
#     Attempts to infer the probability distribution
#     p(z|x) from the data by fitting a variational
#     distribution q_φ(z|x). Returns the two parameters
#     of the distribution (µ, log σ²).
#     :param dims: dimensions of the networks
#         given by the number of neurons on the form
#         [input_dim, [hidden_dims], latent_dim].
#     """
#     def __init__(self, x_dim, z_dim, h_dim, activation=nn.Tanh, distribution=GaussianLayer):
#         super().__init__()
#         # TODO Make batchnormalization optional
#         # TODO Collapse this encoder into the DenseSequentialCoder

#         self.x_dim = x_dim
#         self.z_dim = z_dim
#         self.h_dim = [h_dim] if isinstance(h_dim, int) else h_dim
        
#         dims = [x_dim, *self.h_dim]
#         linear_layers = [nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]
#         activations = [activation() for _ in range(len(linear_layers))]
#         batchnorms = [nn.BatchNorm1d(dims[i]) for i in range(0, len(dims))]

#         self.hidden = nn.ModuleList(linear_layers)
#         self.activations = nn.ModuleList(activations)
#         self.batchnorms = nn.ModuleList(batchnorms)
#         self.sample = distribution(self.h_dim[-1], z_dim)

#     def forward(self, x):
#         for batchnorm, layer, activation in zip(self.batchnorms, self.hidden, self.activations):
#             x = activation(layer(batchnorm(x)))
#         return self.sample(x)


# class LadderDecoder(nn.Module):
#     def __init__(self, dims, activation=nn.Tanh):
#         """
#         The ladder dencoder differs from the standard encoder
#         by using batch-normalization and LReLU activation.
#         Additionally, it also returns the transformation x.
#         :param dims: dimensions of the networks
#             given by the number of neurons on the form
#             [latent_dim, [hidden_dims], input_dim].
#         """
#         super(LadderDecoder, self).__init__()
#         # self.x_dim = x_dim
#         # self.z_dim = z_dim
#         # self.h_dim = h_dim
#         [z_dim, h_dim, x_dim] = dims
#         self.z_dim = z_dim
        
#         import IPython
#         IPython.embed()

#         self.coder1 = LadderEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, distribution=IdentityLayer)
#         # self.linear1 = nn.Linear(x_dim, h_dim)
#         # self.batchnorm1 = nn.BatchNorm1d(h_dim)
#         # self.activation1 = activation()
#         self.merge = GaussianMerge(h_dim, self.z_dim)

#         self.coder2 = LadderEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, distribution=IdentityLayer)
#         # self.linear2 = nn.Linear(x_dim, h_dim)
#         # self.batchnorm2 = nn.BatchNorm1d(h_dim)
#         # self.activation2 = activation()
#         # self.sample = GaussianLayer(h_dim, self.z_dim)

#     def forward(self, x, l_mu=None, l_log_var=None):
#         if l_mu is not None:
#             # Sample from this encoder layer and merge
#             # z = self.linear1(x)
#             # z = self.activation1(self.batchnorm1(z))
#             z = self.coder1(x)
#             q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

#         # Sample from the decoder and send forward
#         # z = self.linear2(x)
#         # z = self.activation2(self.batchnorm2(z))
#         z = self.coder2(x)
#         z, p_mu, p_log_var = self.sample(z)

#         if l_mu is None:
#             return z

#         return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


# class LadderVariationalAutoencoder(nn.Module):
#     def __init__(self, x_dim, z_dim, h_dim):        
#         """
#         Ladder Variational Autoencoder as described by
#         [Sønderby 2016]. Adds several stochastic
#         layers to improve the log-likelihood estimate.
        
#         Args:
#             x_dim (int): Dimensionality of input
#             z_dim (list of int): Dimensionality of the latent stochastic variables
#             h_dim (list of list of int): Dimensionality of each of the hidden layers for each of the latent variables
#         """
#         super().__init__()
#         dims = [x_dim, *z_dim]

#         encoder_layers = [LadderEncoder(x_dim=dims[i - 1], h_dim=h_dim[i - 1], z_dim=dims[i]) for i in range(1, len(dims))]
#         decoder_layers = [LadderDecoder([dims[i - 1], h_dim[i - 1], dims[i]]) for i in range(1, len(h_dim))][::-1]

#         self.encoder = nn.ModuleList(encoder_layers)
#         self.decoder = nn.ModuleList(decoder_layers)
#         self.reconstruction = DenseSequentialCoder(x_dim=x_dim, z_dim=z_dim[0], h_dim=h_dim)
#         self.initialize()

#     def initialize(self):
#         gain = nn.init.calculate_gain('tanh', param=None)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
                    
#     def encode(self, x):
#         latents = []
#         for encoder in self.encoder:
#             x, (z, z_mu, z_log_var) = encoder(x)
#             latents.append((z, z_mu, z_log_var))
#         return latents

#     def decode(self, latents):
#         self.kl_divergence = 0
#         for i, decoder in enumerate([None, *self.decoder]):
#             _, l_mu, l_log_var = latents[i]
#             if i == 0:
#                 # If at top, encoder == decoder, use prior for KL.
#                 z = latents[i][0]
#                 self.kl_divergence += kld_gaussian_gaussian(z, (l_mu, l_log_var))
#             else:
#                 # Perform downward merge of information.
#                 z, kl = decoder(z, l_mu, l_log_var)
#                 self.kl_divergence += kld_gaussian_gaussian(*kl)

#         p_x, p_x_mu, p_x_log_var = self.reconstruction(z)
#         return p_x, p_x_mu, p_x_log_var

#     def forward(self, x):
#         # Gather latent representation from encoders along with final z.
#         latents = self.encode(x)
#         latents = list(reversed(latents))
#         p_x, p_x_mu, p_x_log_var = self.decode(latents)
#         return p_x, p_x_mu, p_x_log_var

#     def sample(self, z):
#         for decoder in self.decoder:
#             z = decoder(z)
#         return self.reconstruction(z)
    

class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        super().__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = MultiLayerPerceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = MultiLayerPerceptron([z_dim, *reversed(h_dim), x_dim], output_activation=F.sigmoid)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        kl = qz * (torch.log(qz + 1e-8) - torch.log(1.0/self.z_dim))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1):
        x = self.encoder(x)

        sample, qz = self.sampler(x, tau)
        self.kl_divergence = kld_gaussian_gaussian(qz)

        x_mu = self.decoder(sample)

        return x_mu

    def sample(self, z):
        return self.decoder(z)
