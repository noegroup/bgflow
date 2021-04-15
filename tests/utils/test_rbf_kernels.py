import pytest
import torch
from bgflow.utils import kernelize_with_rbf, rbf_kernels


def test_kernelize_with_rbf():
    # distance vector  of shape `[n_batch, n_particles, n_particles - 1, 1]`
    distances = torch.tensor([[[[0.], [5]], [[0], [5]], [[5], [5]]]])
    mus = torch.tensor([[[[0., 5]]]])
    gammas = torch.tensor([[[[1., 0.5]]]])
    
    distances2 = torch.tensor([[[[1.]]]], requires_grad=True)
    mus2 = torch.tensor([[[[1., 3]]]])
    gammas2 = torch.tensor([[[[1., 0.5]]]])
    
    rbf1 = torch.exp(- (distances2 - mus2[0, 0, 0, 0]) ** 2 / gammas2[0, 0, 0, 0] ** 2)
    rbf2 = torch.exp(- (distances2 - mus2[0, 0, 0, 1]) ** 2 / gammas2[0, 0, 0, 1] ** 2)
    r1 = rbf1 / (rbf1 + rbf2)
    r2 = rbf2 / (rbf1 + rbf2)

    # Test shapes and math for simple configurations
    rbfs = kernelize_with_rbf(distances, mus, gammas)
    assert torch.allclose(rbfs, torch.tensor([[[[1., 0], [0, 1]], [[1., 0], [0, 1]], [[0, 1], [0, 1]]]]), atol=1e-5)

    rbfs2 = kernelize_with_rbf(distances2, mus2, gammas2)
    assert torch.allclose(rbfs2, torch.cat([r1, r2], dim=-1), atol=1e-5)


def test_rbf_kernels():
    # distance vector  of shape `[n_batch, n_particles, n_particles - 1, 1]`
    distances = torch.tensor([[[[0.], [5]], [[0], [5]], [[5], [5]]]])
    mus = torch.tensor([[[[0., 5]]]])
    gammas = torch.tensor([[[[1., 0.5]]]])
    
    distances2 = torch.tensor([[[[1.]]]], requires_grad=True)
    mus2 = torch.tensor([[[[1., 3]]]])
    gammas2 = torch.tensor([[[[1., 0.5]]]])
    
    rbf1 = torch.exp(- (distances2 - mus2[0, 0, 0, 0]) ** 2 / gammas2[0, 0, 0, 0] ** 2)
    rbf2 = torch.exp(- (distances2 - mus2[0, 0, 0, 1]) ** 2 / gammas2[0, 0, 0, 1] ** 2)
    r1 = rbf1 / (rbf1 + rbf2)
    r2 = rbf2 / (rbf1 + rbf2)
    
    # Test shapes, math and the derivative for simple configurations
    neg_log_gammas = - torch.log(gammas)
    rbfs, derivatives_rbfs = rbf_kernels(distances, mus, neg_log_gammas, derivative=True)
    assert torch.allclose(rbfs, torch.tensor([[[[1., 0], [0, 1]], [[1., 0], [0, 1]], [[0, 1], [0, 1]]]]), atol=1e-5)
    assert torch.allclose(derivatives_rbfs, torch.zeros((1, 3, 2, 2)), atol=1e-5)

    neg_log_gammas2 = - torch.log(gammas2)
    rbfs2, derivatives_rbfs2 = rbf_kernels(distances2, mus2, neg_log_gammas2, derivative=True)
    assert torch.allclose(rbfs2, torch.cat([r1, r2], dim=-1), atol=1e-5)

    # Check derivative of rbf
    dr1 = torch.autograd.grad(r1, distances2, retain_graph=True)
    dr2 = torch.autograd.grad(r2, distances2)
    assert torch.allclose(derivatives_rbfs2, torch.cat([dr1[0], dr2[0]], dim=-1), atol=1e-5)
