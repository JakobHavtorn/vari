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

## Implement KL divergence for Independent distributions (Diagonal Gaussian)
Diagonal Gaussian KL divergences are not implemented in PyTorch ATM. This can be easily achieved by:

Replace 

```python
@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
```

in `torch.distributions.kl` with 

```python
import numbers

def sum_rightmost(value, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    If ``dim`` is 0, no dimensions are summed out.
    If ``dim`` is ``float('inf')``, then all dimensions are summed out.
    If ``dim`` is 1, the rightmost 1 dimension is summed out.
    If ``dim`` is 2, the rightmost two dimensions are summed out.
    If ``dim`` is -1, all but the leftmost 1 dimension is summed out.
    If ``dim`` is -2, all but the leftmost 2 dimensions are summed out.
    etc.
    :param torch.Tensor value: A tensor of ``.dim()`` at least ``dim``.
    :param int dim: The number of rightmost dims to sum out.
    """
    if isinstance(value, numbers.Number):
        return value
    if dim < 0:
        dim += value.dim()
    if dim == 0:
        return value
    if dim >= value.dim():
        return value.sum()
    return value.reshape(value.shape[:-dim] + (-1,)).sum(-1)


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    shared_ndims = min(p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims)
    p_ndims = p.reinterpreted_batch_ndims - shared_ndims
    q_ndims = q.reinterpreted_batch_ndims - shared_ndims
    p = Independent(p.base_dist, p_ndims) if p_ndims else p.base_dist
    q = Independent(q.base_dist, q_ndims) if q_ndims else q.base_dist
    kl = kl_divergence(p, q)
    if shared_ndims:
        kl = sum_rightmost(kl, shared_ndims)
    return kl


@register_kl(Independent, MultivariateNormal)
def _kl_independent_mvn(p, q):
    # if isinstance(p.base_dist, Delta) and p.reinterpreted_batch_ndims == 1:
    #     return -q.log_prob(p.base_dist.v)

    if isinstance(p.base_dist, Normal) and p.reinterpreted_batch_ndims == 1:
        dim = q.event_shape[0]
        p_cov = p.base_dist.scale ** 2
        q_precision = q.precision_matrix.diagonal(dim1=-2, dim2=-1)
        return (0.5 * (p_cov * q_precision).sum(-1)
                - 0.5 * dim * (1 + math.log(2 * math.pi))
                - q.log_prob(p.base_dist.loc)
                - p.base_dist.scale.log().sum(-1))

    raise NotImplementedError
```
