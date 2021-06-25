
import pytest
import torch
from bgflow.nn.flow.pppp import InvertiblePPPP, PPPPScheduler, _iterative_solve
from bgflow.nn.flow.sequential import SequentialFlow


def test_invertible_pppp():
    flow = InvertiblePPPP(2, penalty_parameter=0.1)
    x = torch.tensor([[3.1, -2.0]])
    with torch.no_grad():
        flow.u[:] = torch.tensor([0.4, -0.3])
        flow.v[:] = torch.tensor([0.1, 0.2])
    y, logdet = flow.forward(x)

    x2, logdetinv = flow.forward(y, inverse=True)
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

    # test training mode
    flow.train(False)
    y_test, logdet_test = flow.forward(x)
    assert torch.isclose(y, y_test, atol=1e-5).all()
    assert torch.isclose(logdet, logdet_test, atol=1e-5)
    x_test, logdetinv_test = flow.forward(y, inverse=True)
    assert torch.isclose(x, x_test, atol=1e-5).all()
    assert torch.isclose(-logdet, logdetinv_test, atol=1e-5)


@pytest.mark.parametrize("mode", ["eye", "reverse"])
@pytest.mark.parametrize("dim", [1, 3, 5, 6, 7, 10, 20])
def test_initialization(mode, dim):
    flow = InvertiblePPPP(dim, init=mode)
    assert flow.A.inverse().numpy() == pytest.approx(flow.Ainv.numpy())
    assert torch.det(flow.A).item() == pytest.approx(flow.detA.item())


@pytest.mark.parametrize("order", [2, 3, 7])
def test_iterative_solve(order):
    torch.seed = 0
    a = torch.eye(3)+0.05*torch.randn(3, 3)
    inv = torch.inverse(a) + 0.2*torch.randn(3,3)
    for i in range(10):
        inv = _iterative_solve(a, inv, order)
    assert torch.mm(inv, a).numpy() == pytest.approx(torch.eye(3).numpy(), abs=1e-5)


@pytest.mark.parametrize("model", [InvertiblePPPP(3), SequentialFlow([InvertiblePPPP(3), InvertiblePPPP(3)])])
def test_scheduler(model):
    """API test for a full optimization workflow."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = PPPPScheduler(model, optimizer, n_force_merge=5, n_correct=5, n_recompute_det=5)
    torch.manual_seed(0)
    a = 1e-6*torch.eye(3)
    data = torch.ones(800, 3)
    target = torch.einsum("ij,...j->...i", a, data)
    loss = torch.nn.MSELoss()
    assert loss(data, target) > 1e-1
    for iter in range(100):
        optimizer.zero_grad()
        batch = data[8*iter:8*(iter+1)]
        y, _ = model.forward(batch)
        mse = loss(y, target[8*iter:8*(iter+1)])
        mse_plus_penalty = mse + scheduler.penalty()
        mse_plus_penalty.backward()
        optimizer.step()
        if iter % 10 == 0:
            scheduler.step()
    assert scheduler.penalty().item() >= 0.0
    assert mse < 2e-2  # check that the model has improved
    assert scheduler.i == 10


