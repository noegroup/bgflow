
import pytest
import torch
from torch.distributions import MultivariateNormal, Normal, Independent
from bgflow.distribution import TorchDistribution, NormalDistribution


def _random_mean_cov(dim, device, dtype):
    mean = 10*torch.randn(dim).to(device, dtype)
    # generate a symmetric, positive definite matrix
    cov = torch.triu(torch.rand(dim, dim))
    cov = cov + cov.T + dim * torch.diag(torch.ones(dim))
    cov = cov.to(device, dtype)
    return mean, cov


@pytest.mark.parametrize("dim", (2, 10))
def test_distribution_energy(dim, device, dtype):
    """compare torch's normal distribution with bgflow's normal distribution"""
    n_samples = 7
    mean, cov = _random_mean_cov(dim, device, dtype)
    samples = torch.randn((n_samples, dim)).to(device, dtype)
    normal_trch = TorchDistribution(MultivariateNormal(loc=mean, covariance_matrix=cov))
    normal_bgtrch = NormalDistribution(dim, mean, cov)
    assert torch.allclose(normal_trch.energy(samples), normal_bgtrch.energy(samples), rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("dim", (2, 10))
@pytest.mark.parametrize("sample_shape", (50000, torch.Size([10,1])))
def test_distribution_samples(dim, sample_shape, device, dtype):
    """compare torch's normal distribution with bgflow's normal distribution"""
    mean, cov = _random_mean_cov(dim, device, dtype)
    normal_trch = TorchDistribution(MultivariateNormal(loc=mean, covariance_matrix=cov))
    normal_bgtrch = NormalDistribution(dim, mean, cov)
    samples_trch = normal_trch.sample(sample_shape)
    target_shape = torch.Size([sample_shape]) if isinstance(sample_shape, int) else sample_shape
    assert samples_trch.size() == target_shape + torch.Size([dim])
    if isinstance(sample_shape, int):
        samples_bgtrch = normal_bgtrch.sample(sample_shape)
        # to make sure that both sample from the same distribution, compute divergences
        for p in [normal_trch, normal_bgtrch]:
            for q in [normal_trch, normal_bgtrch]:
                for x in [samples_bgtrch, samples_trch]:
                    for y in [samples_bgtrch, samples_trch]:
                        div = torch.mean(
                            (-p.energy(x) + q.energy(y))
                        )
                        assert torch.abs(div) < 5e-2
