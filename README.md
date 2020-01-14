# vari



## Improvements
- Return value from stochastic sampling layers like GaussianSample and BernoulliSample could be the `torch.distributions` distribution representing that.
  
  This allows using `.sample()`, `.log_prob()` and `torch.distributions.kl.kl_divergence()` methods instead of implementing from scratch.
  It also makes the API of the encoder and decoder the same despite the encoding/decoding distribution.
