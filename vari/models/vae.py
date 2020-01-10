import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from vari.layers import IdentityLayer, GaussianSample, GaussianMerge, GumbelSoftmax
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


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, x_dim, z_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.encoder = DenseSequentialCoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        self.decoder = DenseSequentialCoder(x_dim=z_dim, z_dim=x_dim, h_dim=list(reversed(h_dim)))
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
        z, q_z_mu, q_z_log_var = self.encoder(x)

        # Generative p(x|z)
        x, p_x_mu, p_x_log_var = self.decoder(z)

        # KL Divergence
        self.kl_divergence = kld_gaussian_gaussian(z, (q_z_mu, q_z_log_var))

        return x, p_x_mu, p_x_log_var

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decode(z)


class DeepVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, x_dim, z_dim, h_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        dims = [x_dim, *z_dim]

        encoder_layers = [DenseSequentialCoder(x_dim=dims[i - 1], z_dim=dims[i], h_dim=h_dim) for i in range(1, len(dims))]
        decoder_layers = [DenseSequentialCoder(x_dim=dims[i], z_dim=dims[i - 1], h_dim=h_dim) for i in range(1, len(dims))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)

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
        latents = []
        z = x
        for encoder in self.encoder:
            z, q_z_mu, q_z_log_var = encoder(z)
            latents.append((z, q_z_mu, q_z_log_var))
        return latents

    def decode(self, latents):
        # At top, use prior for KL.
        z, l_mu, l_log_var = latents[0]
        self.kl_divergence = kld_gaussian_gaussian(z, (l_mu, l_log_var))
        for i, decoder in enumerate(self.decoder, start=1):
            # Top-down generative path
            z, p_z_mu, p_z_log_var = decoder(z)
            z, q_z_mu, q_z_log_var = latents[i]
            self.kl_divergence += kld_gaussian_gaussian(z, (q_z_mu, q_z_log_var), (p_z_mu, p_z_log_var))
        x, p_x_mu, p_x_log_var = z, p_z_mu, p_z_log_var
        return x, p_x_mu, p_x_log_var

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
        latents = list(reversed(latents))
        x, p_x_mu, p_x_log_var = self.decode(latents)

        # # KL Divergence
        # self.kl_divergence = kld_gaussian_gaussian(q_z, (q_z_mu, q_z_log_var))

        return x, p_x_mu, p_x_log_var

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        #TODO This should be different from decoder
        return self.decode(z)


class AuxilliaryVariationalAutoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, a_dim=3):
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

        self.aux_encoder = DenseSequentialCoder(x_dim=x_dim, z_dim=a_dim, h_dim=h_dim)
        self.aux_decoder = DenseSequentialCoder(x_dim=x_dim + z_dim, z_dim=a_dim, h_dim=list(reversed(h_dim)))

        self.encoder = DenseSequentialCoder(x_dim=a_dim + x_dim, z_dim=z_dim, h_dim=h_dim)
        self.decoder = DenseSequentialCoder(x_dim=z_dim, z_dim=x_dim, h_dim=list(reversed(h_dim)))
        self.initialize()

    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)
        q_z, q_z_mu, q_z_log_var = self.encoder(torch.cat([x, q_a], dim=1))
        return (q_z, q_z_mu, q_z_log_var), (q_a, q_a_mu, q_a_log_var)

    def decode(self, z):
        p_x, p_x_mu, p_x_log_var = self.decoder(q_z)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([p_x, q_z], dim=1))
        return (p_x, p_x_mu, p_x_log_var), (p_a, p_a_mu, p_a_log_var)

    def forward(self, x):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,x)
        q_z, q_z_mu, q_z_log_var = self.encoder(torch.cat([x, q_a], dim=1))

        # Generative p(x|z)
        p_x, p_x_mu, p_x_log_var = self.decoder(q_z)

        # Generative p(a|z,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([p_x, q_z], dim=1))

        a_kl = kld_gaussian_gaussian(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = kld_gaussian_gaussian(q_z, (q_z_mu, q_z_log_var))
        self.kl_divergence = a_kl + z_kl

        return p_x, p_x_mu, p_x_log_var
    
    def sample(self, z):
        return self.decode(z)


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

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            # Sample from this encoder layer and merge
            z = self.linear1(x)
            z = F.tanh(self.batchnorm1(z))
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.tanh(self.batchnorm2(z))
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


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
            x, (z, z_mu, z_log_var) = encoder(x)
            latents.append((z, z_mu, z_log_var))
        return latents

    def decode(self, latents):
        self.kl_divergence = 0
        for i, decoder in enumerate([None, *self.decoder]):
            _, l_mu, l_log_var = latents[i]
            if i == 0:
                # If at top, encoder == decoder, use prior for KL.
                z = latents[i][0]
                self.kl_divergence += kld_gaussian_gaussian(z, (l_mu, l_log_var))
            else:
                # Perform downward merge of information.
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += kld_gaussian_gaussian(*kl)

        p_x, p_x_mu, p_x_log_var = self.reconstruction(z)
        return p_x, p_x_mu, p_x_log_var

    def forward(self, x):
        # Gather latent representation from encoders along with final z.
        latents = self.encode(x)
        latents = list(reversed(latents))
        p_x, p_x_mu, p_x_log_var = self.decode(latents)
        return p_x, p_x_mu, p_x_log_var

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)


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
