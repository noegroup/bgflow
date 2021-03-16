
import pytest
import torch
from bgtorch import NormalDistribution, ProductDistribution


def test_multi_distribution():
    """Test that a compound normal distribution behaves identical to a multivariate standard normal distribution."""
    n1 = NormalDistribution(3)
    n2 = NormalDistribution(2)
    n3 = NormalDistribution(1)
    n = NormalDistribution(6)
    compound = ProductDistribution([n1, n2, n3], cat_dim=-1)
    n_samples = 10
    samples = compound.sample(n_samples)
    assert samples.shape == n.sample(n_samples).shape
    assert n.energy(samples).shape == compound.energy(samples).shape
    assert n.energy(samples).numpy() == pytest.approx(compound.energy(samples).numpy())


def test_multi_distribution_no_cat():
    """Test that a compound normal distribution behaves identical to a multivariate standard normal distribution."""
    n1 = NormalDistribution(3)
    n2 = NormalDistribution(2)
    n3 = NormalDistribution(1)
    n = NormalDistribution(6)
    compound = ProductDistribution([n1, n2, n3], cat_dim=None)
    n_samples = 10
    samples = compound.sample(n_samples)
    assert len(samples) == 3
    assert isinstance(samples, tuple)
    assert samples[0].shape == (10, 3)
    assert samples[1].shape == (10, 2)
    assert samples[2].shape == (10, 1)
    assert compound.energy(*samples).shape == (10, 1)
    assert n.energy(torch.cat(samples, dim=-1)).numpy() == pytest.approx(compound.energy(*samples).numpy())

