"""
Test inverse and its derivative
"""

import torch
import pytest
from bgflow.nn import flow


@pytest.fixture(params=[
    flow.KroneckerProductFlow(2),
    flow.PseudoOrthogonalFlow(2),
    flow.BentIdentity(),
    flow.FunnelFlow(),
    flow.AffineFlow(2),
    flow.SplitFlow(1),
    flow.SplitFlow(1,1),
    flow.TriuFlow(2),
    flow.InvertiblePPPP(2),
    flow.TorchTransform(torch.distributions.IndependentTransform(torch.distributions.SigmoidTransform(), 1))
])
def simpleflow2d(request):
    return request.param


def test_inverse(simpleflow2d):
    """Test inverse and inverse logDet of simple 2d flow blocks."""
    inverse = flow.InverseFlow(simpleflow2d)
    x = torch.tensor([[1., 2.]])
    *y, dlogp = simpleflow2d._forward(x)
    x2, dlogpinv = inverse._forward(*y)
    assert (dlogp + dlogpinv).detach().numpy() == pytest.approx(0.0, abs=1e-6)
    assert torch.norm(x2 - x).item() == pytest.approx(0.0, abs=1e-6)

    # test dimensions
    assert x2.shape == x.shape
    assert dlogp.shape == x[..., 0, None].shape
    assert dlogpinv.shape == x[..., 0, None].shape

