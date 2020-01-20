
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from vari.layers import Identity, GaussianSample, BernoulliSample, GaussianMerge, GumbelSoftmax
from vari.inference import log_gaussian, log_standard_gaussian
from vari.inference.divergence import kld_gaussian_gaussian, kld_gaussian_gaussian_analytical


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
    Inference network
    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).
    :param dims: dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, x_dim, z_dim, h_dim, activation=nn.Tanh, sample_layer=GaussianSample):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = [h_dim] if isinstance(h_dim, int) else h_dim

        dims = [x_dim, *self.h_dim]
        linear_layers = [nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]
        activations = [activation() for _ in range(len(linear_layers))]

        self.hidden = nn.ModuleList(linear_layers)
        self.activations = nn.ModuleList(activations)
        self.sample = sample_layer(self.h_dim[-1], z_dim)

    def forward(self, x):
        for layer, activation in zip(self.hidden, self.activations):
            x = activation(layer(x))
        return self.sample(x)
    
    
class VariationalInferenceModel(nn.Module):
    """Model that encodes the generative PGM: z_N -> z_N-1 -> ... -> z_1 --> x without amortized inference of z_:.
    """
    def __init__(self):
        raise NotImplementedError()


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, x_dim, z_dim, h_dim, encoder_sample_layer=GaussianSample, decoder_sample_layer=GaussianSample, activation=nn.Tanh, encoder=None, decoder=None):
        super().__init__()

        self.x_dim = x_dim  # if isinstance(x_dim, int) else np.prod(x_dim)  # The dim or flatten
        self.z_dim = z_dim
        self.h_dim = h_dim

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = DenseSequentialCoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, sample_layer=encoder_sample_layer, activation=activation)
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = DenseSequentialCoder(x_dim=z_dim, z_dim=x_dim, h_dim=list(reversed(h_dim)), sample_layer=decoder_sample_layer, activation=activation)
        self.kl_divergences = [0]
        self.kl_divergence = 0
        self.initialize()

    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        # Latent inference q(z|a,x)
        z, pz_args = self.encoder(x)

        # Generative p(x|z)
        x, px_args = self.decoder(z)

        # KL Divergence
        self.kl_divergence = kld_gaussian_gaussian(z, pz_args)
        self.kl_divergences[0] = self.kl_divergence

        return x, px_args

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decode(z)

    def log_likelihood(self, x, *px_args):
        return self.decoder.sample.log_likelihood(x, *px_args)

    def elbo(self, x):
        px, px_args = self.forward(x)
        return self.log_likelihood(x, *px_args) - self.kl_divergence


class HierarchicalVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, x_dim, z_dim, h_dim, encoder_sample_layer, decoder_sample_layer, activation=nn.Tanh):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        assert len(z_dim) <= 2, 'Does not support more than two latents ATM'

        enc_dims = [x_dim, *z_dim]
        dec_dims = enc_dims[::-1]  # reverse
        h_dim_rev = h_dim[::-1]  # reverse

        encoder_layers = [DenseSequentialCoder(x_dim=enc_dims[i - 1], z_dim=enc_dims[i], h_dim=h_dim[i - 1],
                                               sample_layer=encoder_sample_layer[i - 1], activation=activation) for i in range(1, len(enc_dims))]
        decoder_layers = [DenseSequentialCoder(x_dim=dec_dims[i - 1], z_dim=dec_dims[i], h_dim=h_dim_rev[i - 1],
                                               sample_layer=decoder_sample_layer[i - 1], activation=activation) for i in range(1, len(dec_dims))]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)

        self.kl_divergences = [0] * len(z_dim)
        self.kl_divergence = 0
        self.initialize()

    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        """Return list of latents with an element being a tuple of samples and a tuple of the parameters of the q(z|x)
        NOTE Improvement: List or OrderedDict of torch.distributions and then we can sample from this later.
        """
        latents = []
        z = x
        for encoder in self.encoder:
            z, (q_z_mu, q_z_sd) = encoder(z)
            latents.append((z, (q_z_mu, q_z_sd)))
        return latents

    def decode(self, latents, copy_latents=None):
        """Decode a list of latent representations
        
        Args:
            latents (list of tuple): List of samples and distribution parameters [(z1, (z1_parameters)), ...]
            copy_latents (list of bool, optional): If not None must be a list that is True for each latent to copy from.
                                                   the encoder and False when using the generative sample.
                                                   Defaults to None in which case all latents are copied.
        
        Returns:
            tuple: Tuple of samples and distribution parameters for input space [(x, (px_parameters)), ...]
        """
        copy_latents = copy_latents if copy_latents is not None else [True] * len(latents)
        assert len(copy_latents) == len(latents)

        # Top most latent has unconditional prior and is always required to be given (i.e. copied)
        q_z2, (q_z2_mu, q_z2_sd) = latents[-1]
        if copy_latents[-1]:
            self.kl_divergences[1] = kld_gaussian_gaussian(q_z2, (q_z2_mu, q_z2_sd))
            self.kl_divergence = self.kl_divergences[1]
            p_z1, (p_z1_mu, p_z1_sd) = self.decoder[0](q_z2)
        else:
            raise ValueError('copy_latents[-1] must be True since this is the top-most latent that cannot be generated')
    
        q_z1, (q_z1_mu, q_z1_sd) = latents[-2]
        if copy_latents[-2]:
            self.kl_divergences[0] = kld_gaussian_gaussian(q_z1, (q_z1_mu, q_z1_sd), p_param=(p_z1_mu, p_z1_sd))
            self.kl_divergence = self.kl_divergence + self.kl_divergences[0]
            x, px_args = self.decoder[1](q_z1)
        else:
            self.kl_divergences[0] = kld_gaussian_gaussian(p_z1, (p_z1_mu, p_z1_sd), p_param=(q_z1_mu, q_z1_sd))
            self.kl_divergence = self.kl_divergence + self.kl_divergences[0]
            # self.kl_divergences[0] = kld_gaussian_gaussian(p_z1, (q_z1_mu, q_z1_sd), p_param=(p_z1_mu, p_z1_sd))
            # self.kl_divergences[0] = (kld_gaussian_gaussian(q_z1, (q_z1_mu, q_z1_sd), p_param=(p_z1_mu, p_z1_sd)) +
                                    #   kld_gaussian_gaussian(q_z1, (p_z1_mu, p_z1_sd), p_param=(q_z1_mu, q_z1_sd))) / 2
            # self.kl_divergence += self.kl_divergences[1]
            x, px_args = self.decoder[1](p_z1)

        # NOTE THE ABOVE CODE IN CASE WHERE IT IS IN A LOOP
        # q_zi, (q_zi_mu, q_zi_sd) = latents[-2]
        # if copy_latents[-2]:
        #     self.kl_divergences[1] = kld_gaussian_gaussian(q_zi, (q_zi_mu, q_zi_sd), (p_zi_mu, p_zi_sd))
        #     self.kl_divergence += self.kl_divergences[1]
        #     p_zi, (p_zi_mu, p_zi_sd) = self.decoder[1](q_zi)
        # else:
        #     self.kl_divergences[1] = kld_gaussian_gaussian(q_zi, (q_zi_mu, q_zi_sd), p_param=(p_zi_mu, p_zi_sd), p_z=p_zi)
        #     self.kl_divergence += self.kl_divergences[1]
        #     p_zi, (p_zi_mu, p_zi_sd) = self.decoder[1](p_zi)



        # Top most latent has unconditional prior and is always required to be given (i.e. copied)
        # q_z2, (q_z2_mu, q_z2_sd) = latents[-1]
        # self.kl_divergences[0] = kld_gaussian_gaussian(q_z2, (q_z2_mu, q_z2_sd))
        # self.kl_divergence = self.kl_divergences[0]

        # The lower layer latents have priors that are conditional on the previous layer's outupt.
        # If we copy the latent, then we use generative samples to evaluate the log-likelihood of both the posterior,
        # q(zi|zi+1), and of the prior, p(zi|zi-1). If we do not copy, the the samples 
        # for i in range(1, len(self.decoder)):
        #     if copy_latents[-(i+1)]:
        #         self.kl_divergences[i] = kld_gaussian_gaussian(q_zi, qz_args, pz_args)
        #     else:
        #         # If latents are None this signifies that we don't copy the latent encoding from that layer and instead
        #         # use the generated sample
        #         p_zi, pz_args = self.decoder[i - 1](p_zi)
        #         q_zi, qz_args = latents[-(i+1)]
        #         self.kl_divergences[i] = kld_gaussian_gaussian(q_zi, qz_args, pz_args, p_zi)
                
        #     self.kl_divergence += self.kl_divergences[i]

        # if copy_latents[-2]:
        #     x, px_args = self.decoder[-1](q_z1)
        # else:
        #     x, px_args = self.decoder[-1](p_z1)

        return x, px_args

    def forward(self, x):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        # Latent inference q(z|a,x)
        latents = self.encode(x)

        # Generative p(x|z)
        x, px_args = self.decode(latents)

        return x, px_args

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        for decoder in self.decoder:
            z, qz_args = decoder(z)
        return z, qz_args
    
    def log_likelihood(self, x, *px_args):
        return self.decoder[-1].sample.log_likelihood(x, *px_args)


class AuxilliaryVariationalAutoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, a_dim, encoder_sample_layer=GaussianSample, decoder_sample_layer=GaussianSample, activation=nn.Tanh):
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

        self.encoder = DenseSequentialCoder(x_dim=a_dim + x_dim, z_dim=z_dim, h_dim=h_dim, sample_layer=encoder_sample_layer, activation=activation)
        self.decoder = DenseSequentialCoder(x_dim=z_dim, z_dim=x_dim, h_dim=list(reversed(h_dim)), sample_layer=decoder_sample_layer, activation=activation)
        
        self.kl_divergence = 0
        self.kl_divergences = [0] * 2
        self.initialize()

    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
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
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.tanh(self.batchnorm(x))
        return x, self.sample(x)


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
        self.sample = GaussianSample(h_dim, self.z_dim)

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
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
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
#     def __init__(self, x_dim, z_dim, h_dim, activation=nn.Tanh, sample_layer=GaussianSample):
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
#         self.sample = sample_layer(self.h_dim[-1], z_dim)

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

#         self.coder1 = LadderEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, sample_layer=IdentityLayer)
#         # self.linear1 = nn.Linear(x_dim, h_dim)
#         # self.batchnorm1 = nn.BatchNorm1d(h_dim)
#         # self.activation1 = activation()
#         self.merge = GaussianMerge(h_dim, self.z_dim)

#         self.coder2 = LadderEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, sample_layer=IdentityLayer)
#         # self.linear2 = nn.Linear(x_dim, h_dim)
#         # self.batchnorm2 = nn.BatchNorm1d(h_dim)
#         # self.activation2 = activation()
#         # self.sample = GaussianSample(h_dim, self.z_dim)

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
#         gain = torch.nn.init.calculate_gain('tanh', param=None)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.xavier_normal_(m.weight, gain=gain)
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
        k = Variable(torch.FloatTensor([self.z_dim]), requires_grad=False)
        kl = qz * (torch.log(qz + 1e-8) - torch.log(1.0/k))
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
