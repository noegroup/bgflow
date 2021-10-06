import pytest
import torch
import numpy as np
from bgflow.utils import (
    brute_force_jacobian, brute_force_jacobian_trace, get_jacobian,
    batch_jacobian, requires_grad
)


x = torch.tensor([[1., 2, 3]], requires_grad=True)
y = x.pow(2) + x[:, 1]
true_jacobian = np.array([[[2., 1, 0], [0, 5, 0], [0, 1, 6]]])


def test_brute_force_jacobian_trace():
    jacobian_trace = brute_force_jacobian_trace(y, x)
    assert jacobian_trace.detach().numpy() == pytest.approx(np.array([13.]), abs=1e-6)


def test_brute_force_jacobian():
    jacobian = brute_force_jacobian(y, x)
    assert jacobian.detach().numpy() == pytest.approx(true_jacobian, abs=1e-6)


def test_batch_jacobian(ctx):
    t = x.repeat(10, 1).to(**ctx)
    out = t.pow(2) + t[:, [1]]
    expected = torch.tensor(true_jacobian, **ctx).repeat(10, 1, 1)
    assert torch.allclose(batch_jacobian(out, t), expected)


def test_get_jacobian(ctx):
    t = torch.ones((2,), **ctx)
    func = lambda s: s**2
    expected = 2*t*torch.eye(2, **ctx)
    assert torch.allclose(get_jacobian(func, t).jac, expected)


def test_requires_grad(ctx):
    t = torch.zeros((2,), **ctx)
    assert not t.requires_grad
    with requires_grad(t):
        assert t.requires_grad
