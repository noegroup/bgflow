import torch
import numpy as np
from bgflow.nn.flow.spline import PeriodicTabulatedTransform, rq_spline


def test_tabulated_transform():
    """Test that passing the tabulated CDF through the inverse add_transform gives the bin edges"""
    nbins = 10
    n_dofs = 2
    bin_edges = torch.linspace(0., 2 * np.pi, steps=nbins + 1)[None, ...].repeat(n_dofs, 1)

    # mimic the construction in InformedTorsionTransform
    energies = torch.sin(bin_edges * torch.arange(1, 1 + n_dofs)[..., None])
    pdf = torch.exp(-energies)
    trapezoidal = 0.5 * (pdf[..., 1:] + pdf[..., :-1])
    partial_integral = torch.cat([torch.zeros_like(trapezoidal[..., [0]]), torch.cumsum(trapezoidal, axis=-1)], axis=-1)
    pdf /= partial_integral[..., [-1]]
    cdf = partial_integral / partial_integral[..., [-1]]
    trafo = PeriodicTabulatedTransform(
        support_points=bin_edges,
        support_values=cdf,
        slopes=pdf,
    )
    assert torch.allclose(trafo.forward(cdf.T * 0.999999, inverse=True)[0], bin_edges.T, atol=1e-4, rtol=0.0)


def test_rational_quadratic_spline():
    """
    test that spline is exact for sqrt and square
    """
    supportx = torch.stack([torch.arange(0.2, 1.1, 0.2), torch.arange(0.4, 2.1, 0.4)])
    supporty = supportx ** 2
    derivatives = 2 * supportx
    inputs = 0.2 + 0.8 * torch.rand(3, 3, 2)
    inputs[..., 1] *= 2
    outputs, logdet = rq_spline(inputs, supportx, supporty, derivatives)
    outputs_inverse, logdet_inverse = rq_spline(inputs, supportx, supporty, derivatives, inverse=True)
    assert torch.allclose(outputs, inputs ** 2, atol=1e-5, rtol=0)
    assert torch.allclose(outputs_inverse, inputs ** 0.5, atol=1e-5, rtol=0)
    assert torch.allclose(logdet, torch.log(torch.abs(2 * inputs)), atol=1e-5, rtol=0)
    assert torch.allclose(logdet_inverse, torch.log(torch.abs(0.5 * inputs ** -0.5)), atol=1e-5, rtol=0)
    assert torch.allclose(rq_spline(outputs, supportx, supporty, derivatives, inverse=True)[1], -logdet, atol=1e-5,
                          rtol=0)