"""Test spline transformer"""

try:
    import jax
    import jax.numpy as jnp
    from jax.config import config as jax_config
except ImportError:
    jax = None
    jnp = None
    jax_config = None
try:
    import jax2torch
except ImportError:
    jax2torch = None

import contextlib
import functools
import pytest
import numpy as np
import torch
from bgflow.nn.flow.transformer.jax_bridge import (
    bisect,
    chain,
    compose,
    to_torch,
    with_ldj,
    invert_bijector,
    wrap_params,
)
from bgflow.nn.flow.transformer.jax import (
    affine_sigmoid,
    mixture,
    ramp_to_sigmoid,
    remap_to_unit,
    smooth_ramp,
    wrap_around
)


@pytest.mark.skipif(jax is None or jnp is None,
                    reason='skipping test due missing jax installation')
@pytest.mark.skipif(jax2torch is None,
                    reason='skipping test due to missing jax2torch installation')
def test_simple_mixture_transformer(ctx):
    with double_raises(ctx["dtype"]):
        bijector = chain(
            wrap_around,
            mixture,
            affine_sigmoid,
            ramp_to_sigmoid,
            smooth_ramp)
        forward, inverse = to_torch(bijector)

        np.random.seed(43)

        batch = 17
        features = 19
        num_components = 7

        x = torch.from_numpy(np.random.uniform(size=(batch, features))).to(**ctx)
        weights = torch.from_numpy(
            np.random.normal(size=(batch, features, num_components))).to(**ctx)
        shift = torch.from_numpy(
            np.random.normal(size=(batch, features, num_components))).to(**ctx)
        scale = torch.from_numpy(
            np.random.normal(size=(batch, features, num_components))).to(**ctx)
        mix = torch.from_numpy(
            np.random.normal(size=(batch, features, num_components))).to(**ctx)
        logalpha = torch.from_numpy(
            np.random.normal(size=(batch, features, num_components))).to(**ctx)
        y, ldjy = forward(x, (weights, shift, scale, mix, logalpha))
        z, ldjz = inverse(y, (weights, shift, scale, mix, logalpha))
        assert torch.allclose(x, z, rtol=1e-4, atol=1e-4)
        assert torch.allclose(ldjy + ldjz, torch.zeros_like(ldjz),
                              rtol=1e-4, atol=1e-4)


def exp_bijector(x, a, b):
    return jnp.exp(a * x + b)


def exp_bijector_inv(x, a, b):
    return (jnp.log(x) - b) / a


def sin_bijector(x, a, b):
    return jnp.sin(x) * a + b


def sin_bijector_inv(x, a, b):
    return jnp.arcsin((x - b) / a)


def monomial_bijector(x, a, b, power=3):
    return a * jnp.power(x, power) + b


def monomial_bijector_inv(x, a, b, power=3):
    return jnp.power((x - b) / a, 1 / power)


def get_grads(bijector, x, cond):
    (out_y, out_ldj), vjp_fun = jax.vjp(bijector, x, cond)
    vjp_fun = jax.jit(vjp_fun)
    return (vjp_fun((jnp.ones_like(out_y), jnp.zeros_like(out_ldj))),
            vjp_fun((jnp.zeros_like(out_y), jnp.ones_like(out_ldj))))


def abs_err(x, y):
    return jnp.abs(x - y)


def rel_err(x, y, eps=1e-10):
    err = abs_err(x, y)
    denom = jnp.maximum(jnp.abs(y), eps)
    return err / denom


@pytest.mark.skipif(jax is None or jnp is None,
                    reason='skipping test due missing jax installation')
@pytest.mark.skipif(jax2torch is None,
                    reason='skipping test due to missing jax2torch installation')
def test_approx_inv_gradients():
    jax_config.update("jax_enable_x64", True)

    threshold = 1e-6

    bijectors = [exp_bijector, sin_bijector, monomial_bijector]
    inverses = [exp_bijector_inv, sin_bijector_inv, monomial_bijector_inv]

    for fwd, inv in zip(bijectors, inverses):
        fwd = with_ldj(jax.vmap(fwd))
        approx_inv = invert_bijector(fwd, functools.partial(bisect, left_bound=-10, right_bound=10., eps=1e-20))
        inv = with_ldj(jax.vmap(inv))

        fwd = wrap_params(fwd)
        approx_inv = wrap_params(approx_inv)
        inv = wrap_params(inv)

        fwd = jax.jit(fwd)
        approx_inv = jax.jit(approx_inv)
        inv = jax.jit(inv)

        x = jnp.array(np.random.uniform(low=0.1, high=0.9, size=(1000)))
        a = jnp.array(np.random.uniform(low=0.1, high=10., size=(1000)))
        b = jnp.array(np.random.uniform(low=0., high=1., size=(1000)))

        cond = (a, b)

        y, ldjy = fwd(x, cond)
        z, ldjz = inv(y, cond)
        z_, ldjz_ = approx_inv(y, cond)

        (out_y, out_ldj), vjp_fun = jax.vjp(approx_inv, y, cond)

        for label, err_fn in zip(['abs err', 'rel err'], [abs_err, functools.partial(rel_err, eps=threshold)]):
            err_val = np.max(jax.tree_flatten(jax.tree_map(
                jnp.max, jax.tree_multimap(
                    err_fn, get_grads(inv, y, cond), get_grads(approx_inv, y, cond))))[0])
            assert err_val < threshold, f"{label}: {err_val} > {threshold}"


@contextlib.contextmanager
def double_raises(prec):
    if prec == torch.float64:
        with pytest.raises(ValueError, match="currently not supported"):
            yield
    else:
        yield
