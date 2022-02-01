import functools

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


__all__ = [
    'affine_transform',
    'smooth_ramp',
    'monomial_ramp',
    'ramp_to_sigmoid',
    'affine_sigmoid',
    'wrap_around',
    'remap_to_unit',
    'mixture',
]


def affine_transform(x, a, b):
    """Affine transform."""
    return x * jnp.exp(a) + b


def smooth_ramp(x, logalpha, power=1, eps=1e-9):
    """Smooth ramp."""
    assert power > 0
    assert isinstance(power, int)
    assert eps > 0
    alpha = jnp.exp(logalpha)
    # double `where` trick to avoid NaN in backward pass
    z = jnp.where(x > eps, x, jnp.ones_like(x) * eps)
    normalizer = jnp.exp(-alpha * 1.)
    return jnp.where(
        x > eps,
        jnp.exp(-alpha * jnp.power(z, -power)) / normalizer,
        jnp.zeros_like(z))


def monomial_ramp(x, order=2):
    assert order > 0 and isinstance(order, int)
    return jnp.power(x, order)


def ramp_to_sigmoid(ramp):
    """Generalized sigmoid, given a ramp."""
    def _sigmoid(x, *params):
        numer = ramp(x, *params)
        denom = numer + ramp(1. - x, *params)
        return numer / denom
    return _sigmoid


def affine_sigmoid(sigmoid, eps=1e-8):
    """Generalized affine sigmoid transform."""
    assert eps > 0

    def _affine_sigmoid(x, shift, slope, mix, *params):
        slope = jnp.exp(slope)
        mix = jax.nn.sigmoid(mix) * (1. - eps) + eps
        return (mix * sigmoid(slope * (x - shift), *params)
                + (1. - mix) * x)
    return _affine_sigmoid


def wrap_around(bijector, sheaves=None, weights=None):
    """Wraps affine sigmoid around circle."""
    if sheaves is None:
        sheaves = jnp.array([-1, 0, 1])
    if weights is None:
        weights = jnp.zeros_like(sheaves)
    mixture_ = mixture(bijector)

    def _wrapped(x, *params):
        x = x - sheaves[None]
        params = [jnp.repeat(p[..., None], len(sheaves), axis=-1) for p in params]
        return mixture_(x, weights, *params)
    return remap_to_unit(_wrapped)


def remap_to_unit(fun):
    """Maps transformation back to [0, 1]."""
    @functools.wraps(fun)
    def _remapped(x, *params):
        y1 = fun(jnp.ones_like(x), *params)
        y0 = fun(jnp.zeros_like(x), *params)
        return (fun(x, *params) - y0) / (y1 - y0)
    return _remapped


def mixture(bijector):
    """Combines multiple bijectors into a mixture."""
    def _mixture_bijector(x, weights, *params):
        components = jax.vmap(
            functools.partial(bijector, x),
            in_axes=-1,
            out_axes=-1)(*params)
        return jnp.sum(jax.nn.softmax(weights) * components)
    return _mixture_bijector
