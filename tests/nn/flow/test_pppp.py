
import pytest
import torch
from bgtorch.nn.flow.pppp import OrthogonalPPPP, InvertiblePPPP


def test_orthogonal_pppp():
    flow = OrthogonalPPPP(2, penalty_parameter=0.3)
    x = torch.tensor([[3.1,-2.0]])
    flow.Q = torch.tensor([[0.0,1.0],[1.0,0.0]])
    flow.b[:] = torch.tensor([2.1, 1.2])
    dv = torch.tensor([0.1, -0.1])
    flow.dv[:] = dv

    y, logdet = flow._forward(x)
    assert torch.isclose(flow._inverse(y, inverse=True)[0], x, atol=1e-5).all()

    assert(flow.penalty() > 0)
    flow.pppp_merge()
    y2, logdet2 = flow.forward(x)

    assert torch.isclose(flow.forward(y2, inverse=True)[0], x, atol=1e-5).all()
    assert torch.isclose(logdet, torch.zeros(1), atol=1e-5)
    assert torch.isclose(logdet2, torch.zeros(1), atol=1e-5)
    assert torch.isclose(y,y2, atol=1e-5).all()
    assert torch.isclose(flow.dv, torch.zeros(1), atol=1e-5).all()


def test_invertible_pppp():
    flow = InvertiblePPPP(2, penalty_parameter=0.1)
    x = torch.tensor([[3.1, -2.0]])
    flow.u[:] = torch.tensor([0.4, -0.3])
    flow.v[:] = torch.tensor([0.1, 0.2])
    y, logdet = flow.forward(x)

    x2, logdetinv = flow._inverse(y, inverse=True)
    assert torch.isclose(x2, x, atol=1e-5).all()

    assert(flow.penalty() > -1)
    flow.pppp_merge()
    y2, logdet2 = flow.forward(x)
    x3, logdetinv2 = flow._inverse(y2, inverse=True)

    assert torch.isclose(torch.mm(flow.A, flow.Ainv), torch.eye(2)).all()
    assert torch.isclose(x3, x, atol=1e-5).all()
    assert torch.isclose(logdet, -logdetinv, atol=1e-5)
    assert torch.isclose(logdet, logdet2, atol=1e-5)
    assert torch.isclose(logdet, -logdetinv2, atol=1e-5)
    assert torch.isclose(y, y2, atol=1e-5).all()
    assert torch.isclose(flow.u, torch.zeros(1), atol=1e-5).all()
