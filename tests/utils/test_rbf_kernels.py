import pytest
import torch
from bgtorch.utils import kernelize_with_rbf, rbf_kernels

# distance vector  of shape `[n_batch, n_particles, n_particles - 1, 1]`
distances = torch.tensor([[[[0.], [5]], [[0], [5]], [[5], [5]]]])
mus = torch.tensor([[[[0., 5]]]])
gammas = torch.tensor([[[[1., 0.5]]]])


def test_kernelize_with_rbf():
    # Test the input and output shapes for a simple configuration
    # TODO: Test math as well
    rbfs = kernelize_with_rbf(distances, mus, gammas)
    assert torch.allclose(rbfs, torch.tensor([[[[1., 0], [0, 1]], [[1., 0], [0, 1]], [[0, 1], [0, 1]]]]), atol=1e-5)


def test_rbf_kernels():
    # Test the input and output shapes for a simple configuration
    # TODO: Test math as well
    neg_log_gammas = - torch.log(gammas)
    rbfs, derivatives_rbs = rbf_kernels(distances, mus, neg_log_gammas, derivative=True)
    assert torch.allclose(rbfs, torch.tensor([[[[1., 0], [0, 1]], [[1., 0], [0, 1]], [[0, 1], [0, 1]]]]), atol=1e-5)
    assert torch.allclose(derivatives_rbs, torch.zeros((1, 3, 2, 2)), atol=1e-5)
