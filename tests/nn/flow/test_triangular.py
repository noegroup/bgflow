
import torch
import pytest
from bgflow.nn.flow.triangular import TriuFlow


@pytest.mark.parametrize(
    "b", [
        torch.tensor(0.),
        torch.nn.Parameter(torch.tensor([1.,4.,1.])),
        torch.nn.Parameter(torch.zeros(3))
    ])
def test_invert(b):
    tf = TriuFlow(3, shift=(isinstance(b, torch.nn.Parameter)))
    tf._unique_elements = torch.nn.Parameter(torch.rand_like(tf._unique_elements))
    tf._make_r()
    tf.b = b
    x = torch.randn(10,3)
    y, dlogp = tf._forward(x)
    x2, dlogpinv = tf._inverse(y)
    assert torch.norm(dlogp + dlogpinv).item() == pytest.approx(0.0)
    assert torch.norm(x-x2).item() == pytest.approx(0.0, abs=1e-6)
