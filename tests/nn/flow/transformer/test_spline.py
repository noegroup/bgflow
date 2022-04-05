"""Test spline transformer"""

import pytest
import torch
from bgflow import ConditionalSplineTransformer, CouplingFlow, SplitFlow, NormalDistribution, DenseNet


@pytest.mark.parametrize("is_circular", [True, False])
def test_conditional_spline_transformer_api(is_circular, ctx):
    pytest.importorskip("nflows")

    n_bins = 4
    dim_trans = 10
    n_samples = 10
    dim_cond = 9
    x_cond = torch.rand((n_samples, dim_cond), **ctx)
    x_trans = torch.rand((n_samples, dim_trans)).to(x_cond)

    if is_circular:
        dim_net_out = 3 * n_bins * dim_trans
    else:
        dim_net_out = (3 * n_bins + 1) * dim_trans
    conditioner = DenseNet([dim_cond, dim_net_out])

    transformer = ConditionalSplineTransformer(
        params_net=conditioner,
        is_circular=is_circular,
    ).to(x_cond)

    y, dlogp = transformer.forward(x_cond, x_trans)

    assert (y > 0.0).all()
    assert (y < 1.0).all()


@pytest.mark.parametrize(
    "is_circular", [torch.tensor(True), torch.tensor(False), torch.tensor([True, False], dtype=torch.bool)]
)
def test_conditional_spline_continuity(is_circular, ctx):
    pytest.importorskip("nflows")
    torch.manual_seed(2150)

    n_bins = 3
    dim_trans = 2
    n_samples = 1
    dim_cond = 1
    x_cond = torch.rand((n_samples, dim_cond), **ctx)

    if is_circular.all():
        dim_net_out = 3 * n_bins * dim_trans
    elif not is_circular.any():
        dim_net_out = (3 * n_bins + 1) * dim_trans
    else:
        dim_net_out = 3 * n_bins * dim_trans + int(is_circular.sum())
    conditioner = DenseNet([dim_cond, dim_net_out], bias_scale=2.)

    transformer = ConditionalSplineTransformer(
        params_net=conditioner,
        is_circular=is_circular,
    ).to(x_cond)

    slopes = transformer._compute_params(x_cond, dim_trans)[2]
    continuous = torch.isclose(slopes[0,:,0], slopes[0,:,-1]).tolist()
    if is_circular.all():
        assert continuous == [True, True]
    elif not is_circular.any():
        assert continuous == [False, False]
    else:
        assert continuous == [True, False]