# vari


## Dataset likelihoods

For reference and sanity checking, here are some estimates of the maximum attainable log-likelihood for some datasets.

| Dataset                                      | max log p(x) |
| -------------------------------------------- | ------------ |
| Moons (σ=0.00)                               | > 2.6        |
| Moons (σ=0.01)                               | > 1.2        |
| Moons (σ=0.05)                               | > -0.3       |
| Spirals (σ=0.00)                             | > 1.8        |
|                                              |              |
| MNISTBinarized (deterministic at 0.5)        |              |
| MNISTBinarized (dynamic)                     | > -80        |
| MNISTBinarized (static)                      |              |
|                                              |              |
| FashionMNISTBinarized (deterministic at 0.5) |              |
| FashionMNISTBinarized (dynamic)              | > -90        |
| FashionMNISTBinarized (static)               |              |
|                                              |              |
| FashionMNISTContinuous (dynamic)             | > 2500       |
| FashionMNISTContinuous (static)              |              |
| MNISTContinuous (dynamic)                    | > 3400       |
| MNISTContinuous (static)                     |              |


## Improvement projects

- Stochastic (distribution) layers should expect logits as input. I.e. if they have trainable transformations on the input
  they should apply their own activations.

- Make all stochastic (distribution) layers take in the dimensionality of the space (1, 2, ...). Currently only 1D is
  supported but for image outputs it would be easier to have the output space be 2D for images.

- Make importance weighting a wrapper around any model that wraps the `forward` call and first repeats the input iw times

- Add evaluators to make logging on metrics easier and take up less code in experiment files

- Improve experiment setups with configuration files that can be read

- Improve model building to allow more configuration

- Refactor `get_copy_latents` and `copy_latents` argument to decode to be an integer of `free_latents` counted from the lowest to the highest latent.

- LadderVAE at https://github.com/addtt/ladder-vae-pytorch is a useful reference


## Implement analytical KL divergence for Independent distributions (Diagonal Gaussian)
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
