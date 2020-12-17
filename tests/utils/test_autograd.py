import pytest
import torch
import numpy as np
from bgtorch.utils import brute_force_jacobian, brute_force_jacobian_trace

x = torch.tensor([[1., 2, 3]], requires_grad=True)
y = x.pow(2) + x[:, 1]


def test_brute_force_jacobian_trace():
    jacobian_trace = brute_force_jacobian_trace(y, x)
    assert jacobian_trace.detach().numpy() == pytest.approx(np.array([13.]), abs=1e-6)


def test_brute_force_jacobian():
    jacobian = brute_force_jacobian(y, x)
    assert jacobian.detach().numpy() == pytest.approx(np.array([[[2., 1, 0], [0, 5, 0], [0, 1, 6]]]), abs=1e-6)
