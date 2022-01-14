import functools

try:
    import jax
    import jax.numpy as jnp
except:
    jax = None
    jnp = None


def affine_transform(x, a, b):
    """Affine transform."""
    return x * jnp.exp(a) + b


def smooth_ramp(x, power=1, eps=1e-8):
    """Smooth ramp."""
    assert power > 0
    assert eps > 0
    # double `where` trick to avoid NaN in backward pass
    z = jnp.where(x > eps, x, jnp.ones_like(x) * eps)
    return jnp.where(
        x > eps,
        jnp.exp(-jnp.power(z, -power)),
        jnp.zeros_like(z))


def monomial_ramp(x, order=2):
    assert order > 0 and isinstance(order, int)
    return x.pow(order)


def ramp_to_sigmoid(ramp):
    """Generalized sigmoid, given a ramp."""
    def _sigmoid(x):
        return ramp(x) / (ramp(x) + ramp(1. - x))
    return _sigmoid


def affine_sigmoid(sigmoid, eps=1e-8):
    """Generalized affine sigmoid transform."""
    assert eps > 0

    def _affine_sigmoid(x, shift, slope, mix):
        slope = jnp.exp(slope)
        mix = jax.nn.sigmoid(mix) * (1. - eps) + eps
        return (mix * sigmoid(slope * (x - shift))
                + (1. - mix) * x)
    return _affine_sigmoid


def wrap_sigmoid(sigmoid, sheaves=None):
    """Wraps affine sigmoid around circle."""
    if sheaves is None:
        sheaves = jnp.array([-1, 0, 1])
    wrapped = mixture(sigmoid)

    def _wrapped_sigmoid(x, shift, slope, mix):
        weights = jnp.zeros_like(sheaves)
        shift = shift + sheaves
        slope = jnp.tile(slope, [len(sheaves)])
        mix = jnp.tile(mix, [len(sheaves)])
        return wrapped(x, weights, shift, slope, mix)
    return _wrapped_sigmoid


def remap_to_unit(fun):
    """Maps transformation back to [0, 1]."""
    @functools.wraps(fun)
    def _remapped(x, *args):
        y1 = fun(jnp.ones_like(x), *args)
        y0 = fun(jnp.zeros_like(x), *args)
        return (fun(x, *args) - y0) / (y1 - y0)
    return _remapped


def mixture(bijector):
    """Combines multiple bijectors into a mixture."""
    def _mixture_bijector(x, weights, *mixture_conds):
        components = jax.vmap(
            functools.partial(bijector, x),
            in_axes=-1,
            out_axes=-1)(*mixture_conds)
        return jnp.sum(jax.nn.softmax(weights) * components)
    return _mixture_bijector
