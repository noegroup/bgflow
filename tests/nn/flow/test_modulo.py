
import pytest
import torch
from bgflow import IncreaseMultiplicityFlow


@pytest.mark.parametrize('mult', [1, torch.ones(3, dtype=torch.int)])
def test_modulo_flow(ctx, mult):
    m = 3
    mult = m * mult
    x = 1/6 + torch.linspace(0, 1, m + 1)[:-1].view(1, -1).to(**ctx)
    flow = IncreaseMultiplicityFlow(mult).to(**ctx)

    #import pdb;pdb.set_trace()
    x = torch.tile(x[None, ...], (1000, 1))
    y, dlogp = flow.forward(x, inverse=True)
    assert torch.allclose(y, 1/2 * torch.ones_like(y))

    x2, dlogp2 = flow.forward(y)
    for point in (1/6, 3/6, 5/6):
        count = torch.sum(torch.isclose(x2, point * torch.ones_like(x2)))
        assert count > 800
        assert count < 1200
    assert torch.allclose(dlogp, -dlogp2)


