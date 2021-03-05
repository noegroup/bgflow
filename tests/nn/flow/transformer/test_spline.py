"""Test spline transformer"""

import pytest
import torch
from bgtorch import ConditionalSplineTransformer, CouplingFlow, SplitFlow, NormalDistribution, DenseNet


@pytest.mark.parametrize("is_circular", [True, False])
def test_neural_spline_flow(is_circular, device, dtype):

    n_bins = 4
    dim_trans = 10
    n_samples = 10
    dim_cond = 9
    x_cond = torch.rand((n_samples, dim_cond), device=device, dtype=dtype)
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