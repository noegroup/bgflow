"""
adapted from
https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py
(MIT License)
"""

import torch
from bgtorch import rational_quadratic_spline, unconstrained_rational_quadratic_spline


def test_constrained_spline_forward_inverse_are_consistent():
    num_bins = 10
    shape = [2,3,4]

    unnormalized_widths = torch.randn(*shape, num_bins)
    unnormalized_heights = torch.randn(*shape, num_bins)
    unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

    def call_spline_fn(inputs, inverse=False):
        return rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse
        )

    inputs = torch.rand(*shape)
    outputs, logabsdet = call_spline_fn(inputs, inverse=False)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

    assert torch.allclose(inputs, inputs_inv, atol=1e-4, rtol=0.0)
    assert torch.allclose(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet), atol=1e-4, rtol=0.0)


def test_uncostrained_spline_forward_inverse_are_consistent():
    num_bins = 10
    shape = [2,3,4]

    unnormalized_widths = torch.randn(*shape, num_bins)
    unnormalized_heights = torch.randn(*shape, num_bins)
    unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

    def call_spline_fn(inputs, inverse=False):
        return unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse
        )

    inputs = 3 * torch.randn(*shape) # Note inputs are outside [0,1].
    outputs, logabsdet = call_spline_fn(inputs, inverse=False)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

    assert torch.allclose(inputs, inputs_inv, atol=1e-4, rtol=0.0)
    assert torch.allclose(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet), atol=1e-4, rtol=0.0)