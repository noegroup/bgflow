"""Test spline transformer"""

import torch
from bgflow.nn.flow.transformer.jax_bridge import (
    chain,
    compose,
    to_bgflow
)
from bgflow.nn.flow.transformer.jax import (
    affine_sigmoid,
    mixture,
    ramp_to_sigmoid,
    remap_to_unit,
    smooth_ramp
)


def test_simple_mixture_transformer():
    bijector = chain(
        mixture,
        remap_to_unit,
        affine_sigmoid,
        ramp_to_sigmoid,
        smooth_ramp)
    forward, inverse = to_bgflow(bijector)

    batch = 128
    features = 64
    num_components = 16

    x = torch.rand(batch, features)
    weights = torch.randn(batch, features, num_components)
    shift = torch.randn(batch, features, num_components)
    scale = torch.randn(batch, features, num_components)
    mix = torch.randn(batch, features, num_components)
    y, ldjy = forward(x, (weights, shift, scale, mix))
    z, ldjz = inverse(y, (weights, shift, scale, mix))
    assert torch.allclose(x, z, rtol=1e-4, atol=1e-3)
    assert torch.allclose(ldjy + ldjz, torch.zeros_like(ldjz),
                          rtol=1e-4, atol=1e-3)
