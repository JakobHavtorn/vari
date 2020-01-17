# vari



## Improvements

- Return value from stochastic sampling layers like GaussianSample and BernoulliSample could be the `torch.distributions` distribution representing that.

  - This allows using `.rsample()`, `.log_prob()` and `torch.distributions.kl.kl_divergence()` methods instead of implementing from scratch.
  - It also makes the API of the encoder and decoder the same despite the encoding/decoding distribution.

- Move ELBO computation into model with (no_grad, importance_samples) as options

- Define encoder and decoder outside VAE model and pass as arguments to enable convolutional coders

- Share some experiment code across experiments (like getting model and datasets from names)

- Refactor minor things

  - Make parameters be a Sequence model in the BernoulliSample class

- (Implement importance sampling as sampling additional samples from the prior compared to input and computing likelihood on repeated inputs (instead of forward passing copies))
