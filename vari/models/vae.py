import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from vari.layers import GaussianSample, GaussianMerge, GumbelSoftmax
from vari.inference import log_gaussian, log_standard_gaussian


# TODO Collapse the LadderEncoder and LadderDecoder into the Encoder and Decoder into single classes
#      and make the LadderVariationalAutoEncoder use these instead


def kld_gaussian_gaussian(z, q_param, p_param=None):
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

    if p_param is None:
        pz = log_standard_gaussian(z)
    else:
        (mu, log_var) = p_param
        pz = log_gaussian(z, mu, log_var)

    kl = qz - pz

    return kl


class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
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


class Encoder(nn.Module):
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
    def __init__(self, x_dim, z_dim, h_dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dims = h_dims

        dims = [x_dim, *h_dims]
        linear_layers = [nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dims[-1], z_dim)
        self.initialize()
        
    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for layer in self.hidden:
            x = F.tanh(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    """
    Generative network
    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).
    :param dims: dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """
    def __init__(self, x_dim, z_dim, h_dims, activation=nn.Tanh, sample_layer=GaussianSample):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dims = h_dims

        dims = [z_dim, *h_dims]
        linear_layers = [nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]
        #activations = [activation() for i in range(1, len(dims))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dims[-1], x_dim)
        self.initialize()
        
    def initialize(self):
        gain = torch.nn.init.calculate_gain('tanh', param=None)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for layer in self.hidden:
            x = F.tanh(layer(x))
        return self.sample(x)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder [Kingma 2013] model
    consisting of an encoder/decoder pair for which
    a variational distribution is fitted to the
    encoder. Also known as the M1 model in [Kingma 2014].
    :param dims: x, z and hidden dimensions of the networks
    """
    def __init__(self, x_dim, z_dim, h_dims):
        super(VariationalAutoencoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dims = h_dims

        self.encoder = Encoder(x_dim=x_dim, z_dim=z_dim, h_dims=h_dims)
        self.decoder = Decoder(x_dim=z_dim, z_dim=x_dim, h_dims=list(reversed(h_dims)))
        self.kl_divergence = 0

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        # Latent inference q(z|a,x)
        q_z, q_z_mu, q_z_log_var = self.encoder(x)

        # Generative p(x|z)
        p_x, p_x_mu, p_x_log_var = self.decoder(q_z)

        # KL Divergence
        self.kl_divergence = kld_gaussian_gaussian(q_z, (q_z_mu, q_z_log_var))

        return p_x, p_x_mu, p_x_log_var

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class AuxilliaryVariationalAutoencoder(nn.Module):
    def __init__(self, x_dim, z_dim, a_dim, h_dims):
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
        self.h_dims = h_dims

        self.aux_encoder = Encoder(x_dim=x_dim, z_dim=a_dim, h_dims=h_dims)
        self.aux_decoder = Encoder(x_dim=x_dim + z_dim, z_dim=a_dim, h_dims=list(reversed(h_dims)))

        self.encoder = Encoder(x_dim=a_dim + x_dim, z_dim=z_dim, h_dims=h_dims)
        self.decoder = Decoder(x_dim=z_dim, z_dim=x_dim, h_dims=list(reversed(h_dims)))

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

        # Generative p(a|q_z,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([p_x, q_z], dim=1))

        a_kl = kld_gaussian_gaussian(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = kld_gaussian_gaussian(q_z, (q_z_mu, q_z_log_var))

        self.kl_divergence = a_kl + z_kl

        return p_x, p_x_mu, p_x_log_var
    

class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        super(GumbelAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Perceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), x_dim], output_activation=F.sigmoid)

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


class LadderEncoder(nn.Module):
    """
    The ladder encoder differs from the standard encoder
    by using batch-normalization and LReLU activation.
    Additionally, it also returns the transformation x.
    :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
    """
    def __init__(self, dims):
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
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
            z = F.leaky_relu(self.batchnorm1(z), 0.1)
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.leaky_relu(self.batchnorm2(z), 0.1)
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self, dims):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.
        :param dims: x, z and hidden dimensions of the networks
        """
        [x_dim, z_dim, h_dim] = dims
        super(LadderVariationalAutoencoder, self).__init__([x_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0], h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for encoder in self.encoder:
            x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            # If at top, encoder == decoder,
            # use prior for KL.
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += kld_gaussian_gaussian(z, (l_mu, l_log_var))

            # Perform downword merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += kld_gaussian_gaussian(*kl)

        x_mu = self.reconstruction(z)
        return x_mu

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)