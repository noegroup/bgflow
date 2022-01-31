
from functools import partial
import numpy as np
import pytest
import torch
from bgflow.nn.flow.piecewise import (
    piecewise_transform, linear_piece, piecewise_rational_quadratic,
    piecewise_linear, bucketize_batch, PiecewiseLinearTransform, PeriodicPiecewiseRationalQuadraticTransform
)


def test_batch_bucketize():
    t = lambda s: torch.tensor(s)[None, None, ...]
    edges = t([0.0, 0.5, 1.0])[None, ...]
    e = edges.clone()
    for x, i in (
        (-0.0001, -1),
        (0.0, 0),
        (0.25, 0),
        (0.5, 1),
        (0.75, 1),
        (1.0, 1),
        (1.0001, 2)
    ):
        assert bucketize_batch(t(x), edges) == t(i)
    assert (edges == e).all()


def simple_linear(ctx):
    bin_edges = torch.tensor([0.0, 0.5, 1.0], **ctx)
    slope = torch.tensor([1.0, 2.0], **ctx)
    return partial(
        piecewise_transform,
        bin_edges=bin_edges,
        bin_transform=linear_piece,
        slope=slope,
        x0=bin_edges[..., :-1],
        y0=torch.tensor([0.0, 1.0], **ctx)
    ), slope, bin_edges


def test_piecewise_transform(ctx):
    function, slope, bin_edges = simple_linear(ctx)
    z = torch.linspace(0.0, 1.0, 99).to(slope)
    bin0 = (z < 0.5)
    bin1 = (z >= 0.5)
    y, dlogp = function(z)
    assert torch.allclose(y[bin0], z[bin0])
    assert torch.allclose(y[bin1], slope[1]*(z[bin1] - 0.5) + 1.0)
    assert torch.allclose(dlogp[bin0], torch.zeros_like(dlogp[bin0]))
    assert torch.allclose(dlogp[bin1], np.log(2) * torch.ones_like(dlogp[bin1]))


@pytest.mark.parametrize("trafo_type", ["rqs", "linear"])
def test_zero_yields_identity(trafo_type):
    batch_size = 12
    dim = 10
    n_bins = 10
    w = torch.zeros(dim, n_bins)
    h = torch.zeros(dim, n_bins)
    s = torch.zeros(dim, n_bins + 1)
    z = torch.rand(batch_size, dim)
    trafos = {
        "rqs": piecewise_rational_quadratic(w, h, s),
        "linear": piecewise_linear(w, h)
    }
    forward_inverse = trafos[trafo_type]
    for trafo in forward_inverse:
        x, dlogp = trafo(z)
        assert torch.allclose(x, z)
        assert torch.allclose(dlogp, torch.zeros_like(dlogp), atol=1e-5)


@pytest.mark.parametrize("trafo_type", ["rqs", "linear"])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("n_bins", [1, 20])
def test_rational_quadratic(trafo_type, temperature, n_bins):
    batch_size = 2
    dim = 2
    w = torch.randn(dim, n_bins)
    h = torch.randn(dim, n_bins)
    s = torch.randn(dim, n_bins + 1)
    trafos = {
        "rqs": piecewise_rational_quadratic(w, h, s, temperature),
        "linear": piecewise_linear(w, h, temperature)
    }
    forward, inverse = trafos[trafo_type]
    z = torch.rand(batch_size, dim)
    x, dlogp = forward(z)
    z2, dlogp2 = inverse(x)
    assert torch.allclose(z, z2, atol=1e-4)
    assert torch.allclose(dlogp, -dlogp2, atol=1e-4)


def test_out_of_bounds(ctx):
    function, slope, bin_edges = simple_linear(ctx)
    with pytest.raises(AssertionError):
        function(torch.tensor([-1.0]))
    # TODO make this more general


@pytest.mark.parametrize("TransformType", [PiecewiseLinearTransform, PeriodicPiecewiseRationalQuadraticTransform])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_transforms(ctx, TransformType, temperature):
    transform = TransformType(dim=4, n_bins=12)
    transform.to(**ctx)
    z = torch.rand(10, 4, **ctx)
    x, dlogp = transform.forward(z, temperature=temperature)
    z2, dlogp2 = transform.forward(x, temperature=temperature, inverse=True)

    atol = {torch.float32: 1e-4, torch.float64: 1e-8}[ctx["dtype"]]
    assert torch.allclose(z, z2, atol=atol)
    atol = {torch.float32: 1e-3, torch.float64: 1e-8}[ctx["dtype"]]
    assert torch.allclose(dlogp, -dlogp2, atol=atol)



@pytest.mark.parametrize("TransformType", [PiecewiseLinearTransform, PeriodicPiecewiseRationalQuadraticTransform])
def test_temperature_scaling(ctx, TransformType):
    transform = TransformType(dim=4, n_bins=12)
    transform.to(**ctx)
    x = torch.rand(10, 4, **ctx)
    z1, dlogp = transform.forward(x, temperature=1.0, inverse=True)
    z2, dlogp2 = transform.forward(x, temperature=2.0, inverse=True)
    boltzmann_indicator = dlogp - 1/2 *  dlogp2
    print(boltzmann_indicator)
    mean = boltzmann_indicator.mean()
    assert torch.allclose(boltzmann_indicator, mean * torch.ones_like(boltzmann_indicator))



def test_temperature_tensor():
    transform = TransformType(dim=4, n_bins=12)
    transform.to(**ctx)
    z = torch.rand(10, 4, **ctx)
    temperature = 2*torch.rand(10, 1)
    x, dlogp = transform.forward(z, temperature=temperature)
    z2, dlogp2 = transform.forward(x, temperature=temperature, inverse=True)
