"""Test spline transformer"""

try:
    import jax
except ImportError:
    jax = None
try:
    import jax2torch
except ImportError:
    jax2torch = None

import pytest
import numpy as np
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


@pytest.mark.skipif(jax is None,
                    reason='skipping test due missing jax installation')
@pytest.mark.skipif(jax2torch is None,
                    reason='skipping test due to missing jax2torch installation')
def test_simple_mixture_transformer():
    bijector = chain(
        mixture,
        remap_to_unit,
        affine_sigmoid,
        ramp_to_sigmoid,
        smooth_ramp)
    forward, inverse = to_bgflow(bijector)

    np.random.seed(43)

    batch = 17
    features = 19
    num_components = 7

    x = torch.from_numpy(np.random.uniform(size=(batch, features))).float()
    weights = torch.from_numpy(
        np.random.normal(size=(batch, features, num_components))).float()
    shift = torch.from_numpy(
        np.random.normal(size=(batch, features, num_components))).float()
    scale = torch.from_numpy(
        np.random.normal(size=(batch, features, num_components))).float()
    mix = torch.from_numpy(
        np.random.normal(size=(batch, features, num_components))).float()
    y, ldjy = forward(x, (weights, shift, scale, mix))
    z, ldjz = inverse(y, (weights, shift, scale, mix))
    assert torch.allclose(x, z, rtol=1e-4, atol=1e-4)
    assert torch.allclose(ldjy + ldjz, torch.zeros_like(ldjz),
                          rtol=1e-4, atol=1e-4)
