
import pytest
import torch
from bgflow import IncreaseMultiplicityFlow
from bgflow import CircularShiftFlow


@pytest.mark.parametrize('mult', [1, torch.ones(3, dtype=torch.int)])
def test_IncreaseMultiplicityFlow(ctx, mult):
    m = 3
    mult = m * mult
    flow = IncreaseMultiplicityFlow(mult).to(**ctx)

    x = 1/6 + torch.linspace(0, 1, m + 1)[:-1].to(**ctx)
    x = torch.tile(x[None, ...], (1000, 1))
    y, dlogp = flow.forward(x, inverse=True)
    assert y.shape == x.shape
    assert torch.allclose(y, 1/2 * torch.ones_like(y))

    x2, dlogp2 = flow.forward(y)
    for point in (1/6, 3/6, 5/6):
        count = torch.sum(torch.isclose(x2, point * torch.ones_like(x2)))
        assert count > 800
        assert count < 1200
    assert torch.allclose(dlogp, torch.zeros_like(dlogp))
    assert torch.allclose(dlogp2, torch.zeros_like(dlogp))

@pytest.mark.parametrize('shift', [1, torch.ones(3)])
def test_CircularShiftFlow(ctx, shift):
    m = 0.2
    shift = m * shift
    flow = CircularShiftFlow(shift).to(**ctx)

    x = torch.tensor([0.0, 0.2, 0.9]).to(**ctx)
    x_shifted = torch.tensor([0.2, 0.4, 0.1]).to(**ctx)
    y, dlogp = flow.forward(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x_shifted)

    x2, dlogp2 = flow.forward(y, inverse=True)
    assert torch.allclose(x, x2)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp))
    assert torch.allclose(dlogp2, torch.zeros_like(dlogp))
