
import pytest
import numpy as np
import torch
from torch.nn.functional import relu
from torch.distributions import Normal, Uniform, AffineTransform
from bgflow.distribution import TorchDistribution, ProductDistribution
from bgflow import (
    jensen_shannon_divergence, JensenShannonDivergence,
    BoltzmannGenerator, TorchTransform, Energy, Sampler,
    SequentialFlow, MergeFlow, SplitFlow
)
from bgflow.utils import pack_tensor_in_tuple


@pytest.mark.parametrize("equal_distributions", [True, False])
@pytest.mark.parametrize("product", [True, False])
def test_jensen_shannon_function_normals(ctx, equal_distributions, product):
    """For normal distributions, we just check the bounds."""
    p, q = _normals(ctx, product)
    if equal_distributions:
        p = q
    xp = pack_tensor_in_tuple(p.sample(1000))
    xq = pack_tensor_in_tuple(q.sample(1000))

    js = jensen_shannon_divergence(
        p.energy(*xp), p.energy(*xq),
        q.energy(*xp), q.energy(*xq),
        free_energy_ua_to_ub=0.0
    )
    if equal_distributions:
        assert js.abs() < 1e-5
    else:
        assert js > 0.15
        assert js < np.log(2)


@pytest.mark.parametrize(
    "i_a_b",
    (
            (0, (-0.5, 0.5), (0.0, 1.0)),
            (1, (-0.5, 0.5), (-0.5, 0.5)),
            (2, (-0.5, 0.5), (0.5, 1.5)),
            (3, (-0.5, 100.), (25.0, 50.0)),
    )
)
@pytest.mark.parametrize("product", [True, False])
def test_jensen_shannon_function_uniforms(ctx, i_a_b, product):
    """For uniform distributions, we compare with an analytic solution."""
    i, a, b = i_a_b
    low1, high1 = a[0] * torch.ones(2, **ctx), a[1] * torch.ones(2, **ctx)
    low2, high2 = b[0] * torch.ones(2, **ctx), b[1] * torch.ones(2, **ctx)
    p, q = _uniforms(low1, high1, low2, high2, product)

    xp = pack_tensor_in_tuple(p.sample(1000))
    xq = pack_tensor_in_tuple(q.sample(1000))

    js = jensen_shannon_divergence(
        p.energy(*xp), p.energy(*xq),
        q.energy(*xp), q.energy(*xq),
        free_energy_ua_to_ub=0.0
    )

    if i == 1:
        assert torch.allclose(js, torch.zeros_like(js))
    elif i == 2:
        assert torch.allclose(js, np.log(2)*torch.ones_like(js))

    assert torch.allclose(js, _js_uniform_analytic(low1, high1, low2, high2), atol=0.05)


@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize(
    "i_shift_scale", (
        (0, 0.0, 1.0),
        (0, 0.3, 1.2),
    )
)
def test_jensen_shannon_class(product, i_shift_scale, ctx):
    i, shift, scale = i_shift_scale
    low, high = torch.zeros(2, **ctx), torch.ones(2, **ctx)
    p, q = _uniforms(
        low, high,
        low, high,
        product
    )
    flow = TorchTransform(
        AffineTransform(loc=shift*torch.ones(2, **ctx), scale=scale*torch.ones(2, **ctx)),
        reinterpreted_batch_ndims=1
    )
    if product:
        flow = SequentialFlow([MergeFlow(1, 1), flow, SplitFlow(1, 1)])
    class Target(Energy, Sampler):
        def __init__(self, delegate, free_energy):
            super().__init__(delegate.event_shapes)
            self.delegate = delegate
            self.free_energy = free_energy
        def _energy(self, *xs, **kwargs):
            return self.free_energy + self.delegate.energy(*xs, **kwargs)
        def _sample(self, n_samples, *args, **kwargs):
            return self.delegate.sample(n_samples, *args, **kwargs)
    target = Target(p, free_energy=torch.tensor(3.14, **ctx))
    generator = BoltzmannGenerator(prior=q, flow=flow, target=target)
    jsdiv = JensenShannonDivergence(generator, target=target)
    jsdiv.update_free_energy(2000)
    assert torch.allclose(jsdiv.target_free_energy, target.free_energy, atol=0.15)
    for i in range(2):
        js = jsdiv(1000, update_free_energy=True)
        assert torch.allclose(js, _js_uniform_analytic(low, high, shift+scale*low, shift+scale*high), atol=0.05)
    js, res = jsdiv(100, return_result_dict=True)
    assert isinstance(res["xref"], tuple)
    assert isinstance(res["xgen"], tuple) or isinstance(res["xgen"], list)
    assert res["uref_on_xref"].shape[0] == res["xref"][0].shape[0]
    assert res["ugen_on_xref"].shape[0] == res["xref"][0].shape[0]


def _make_normal(mu, sigma, dim, **ctx):
    return TorchDistribution(
        Normal(loc=mu * torch.ones(dim, **ctx), scale=sigma * torch.ones(dim, **ctx)),
        reinterpreted_batch_ndims=1
    )


def _make_uniform(low, high):
    return TorchDistribution(
        Uniform(low=low, high=high, validate_args=False),
        reinterpreted_batch_ndims=1,
    )


def _js_uniform_analytic(low1, high1, low2, high2):
    apq = _union_area(low1, high1, low2, high2)
    ap = _area(low1, high1)
    aq = _area(low2, high2)
    i_plnp = -torch.log(ap)
    i_qlnq = -torch.log(aq)
    i_plnm = apq/ap * torch.log(0.5/ap + 0.5/aq) + (ap - apq)/ap * torch.log(0.5/ap)
    i_qlnm = apq/aq * torch.log(0.5/ap + 0.5/aq) + (aq - apq)/aq * torch.log(0.5/aq)
    return 0.5*(i_plnp - i_plnm + i_qlnq - i_qlnm)


def _area(low, high):
    return torch.prod(relu(high - low))


def _union_area(low1, high1, low2, high2):
    return _area(torch.maximum(low1, low2), torch.minimum(high1, high2))


def _uniforms(low1, high1, low2, high2, product=False):
    if product:
        p = ProductDistribution([
            _make_uniform(low1[:1], high1[:1]),
            _make_uniform(low1[1:], high1[1:]),
        ])
        q = ProductDistribution([
            _make_uniform(low2[:1], high2[:1]),
            _make_uniform(low2[1:], high2[1:]),
        ])
    else:
        p = _make_uniform(low1, high1)
        q = _make_uniform(low2, high2)
    return p, q


def _normals(ctx, product):
    if product:
        p = ProductDistribution([
            _make_normal(0.0, 1.0, 1, **ctx),
            _make_normal(0.0, 1.0, 1, **ctx),
        ])
        q = ProductDistribution([
            _make_normal(0.5, 2.0, 1, **ctx),
            _make_normal(0.5, 2.0, 1, **ctx),
        ])
    else:
        p = _make_normal(0.0, 1.0, 2, **ctx)
        q = _make_normal(0.5, 2.0, 2, **ctx)
    return p, q
