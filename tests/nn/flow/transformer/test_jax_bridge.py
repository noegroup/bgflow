"""Test spline transformer"""


try:
    import jax
    import jax.numpy as jnp
    from jax.config import config as jax_config
except ImportError:
    jax = None
    jnp = None
    jax_config = None

import contextlib
import functools
import pytest
import numpy as np
import torch
from bgflow.nn.flow.transformer.jax_bridge import (
    bisect,
    chain,
    to_torch,
    with_ldj,
    invert_bijector,
    JaxTransformer
)
from bgflow.nn.flow.transformer.jax import (
    affine_sigmoid,
    mixture,
    ramp_to_sigmoid,
    smooth_ramp,
    wrap_around
)


@pytest.mark.skipif(jax is None or jnp is None,
                    reason='skipping test due missing jax installation')
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


def get_grads(bijector, x, *cond):
    (out_y, out_ldj), vjp_fun = jax.vjp(bijector, x, *cond)
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
def test_approx_inv_gradients():
    jax_config.update("jax_enable_x64", True)

    threshold = 1e-6

    bijectors = [exp_bijector, sin_bijector, monomial_bijector]
    inverses = [exp_bijector_inv, sin_bijector_inv, monomial_bijector_inv]

    for fwd, inv in zip(bijectors, inverses):
        fwd = with_ldj(jax.vmap(fwd))
        approx_inv = invert_bijector(fwd, functools.partial(bisect, left_bound=-10, right_bound=10., eps=1e-20))
        inv = with_ldj(jax.vmap(inv))

        fwd = jax.jit(fwd)
        approx_inv = jax.jit(approx_inv)
        inv = jax.jit(inv)

        x = jnp.array(np.random.uniform(low=0.1, high=0.9, size=(1000)))
        a = jnp.array(np.random.uniform(low=0.1, high=10., size=(1000)))
        b = jnp.array(np.random.uniform(low=0., high=1., size=(1000)))

        y, ldjy = fwd(x, a, b)
        z, ldjz = inv(y, a, b)
        z_, ldjz_ = approx_inv(y, a, b)

        (out_y, out_ldj), vjp_fun = jax.vjp(approx_inv, y, a, b)

        for label, err_fn in zip(['abs err', 'rel err'], [abs_err, functools.partial(rel_err, eps=threshold)]):
            err_val = np.max(jax.tree_util.tree_flatten(jax.tree_util.tree_map(
                jnp.max, jax.tree_util.tree_map(
                    err_fn, get_grads(inv, y, a, b), get_grads(approx_inv, y, a, b))))[0])
            assert err_val < threshold, f"{label}: {err_val} > {threshold}"

    jax_config.update("jax_enable_x64", False)


@pytest.mark.skipif(jax is None or jnp is None,
                    reason='skipping test due missing jax installation')
def test_bgflow_interface(ctx):
    with double_raises(ctx["dtype"]):
        dimx = 2
        dimy = 2
        num_mixtures = 7
        num_params = 4

        net = torch.nn.Sequential(
            torch.nn.Linear(dimx, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_params * dimy * num_mixtures),
        ).to(**ctx)

        def compute_params(x, y_shape):
            params = net(x).chunk(num_params, dim=-1)
            return tuple(p.view(*p.shape[:-1], y_shape[-1], num_mixtures)
                         for p in params)

        transformer = JaxTransformer(
            chain(
                functools.partial(wrap_around, sheaves=jnp.linspace(-10, 10, 21), weights=jnp.exp(-jnp.linspace(-10, 10, 21))),
                mixture,
                affine_sigmoid,
                jax.nn.tanh
            ),
            compute_params,
            bisection_eps=1e-20
        ).to(**ctx)

        print(ctx)
        x = torch.rand(103, dimx).to(**ctx)
        y = torch.rand(103, dimy).to(**ctx)
        print(x.dtype)

        y1, ldj1 = transformer(x, y, inverse=False)
        print(y1.dtype)
        y2, ldj2 = transformer(x, y1, inverse=True)
        print(y2.dtype)

        assert torch.allclose(y, y2, atol=1e-5, rtol=1e-3), (y - y2).abs().max()
        assert torch.allclose(ldj1, -ldj2, atol=1e-5, rtol=1e-3), (ldj1 + ldj2).abs().max()


@contextlib.contextmanager
def double_raises(prec):
    if prec == torch.float64:
        with pytest.raises(ValueError, match="currently not supported"):
            yield
    else:
        yield
