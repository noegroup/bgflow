import torch
import pytest
import numpy as np
from bgtorch.distribution import NormalDistribution, TruncatedNormalDistribution


@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize("temperature", [1.0, 0.5, 2])
def test_normal_distribution(dim, n_samples, temperature):
    """Test sampling of the Normal Distribution."""

    cov = torch.tensor([[1, 0.3], [0.3, 2]])
    mean = torch.ones(dim)
    normal_distribution = NormalDistribution(dim, mean=mean, cov=cov)

    samples = normal_distribution.sample(n_samples, temperature=temperature)
    tol = 0.1 * np.sqrt(temperature)
    assert samples.shape == torch.Size([n_samples, dim])
    assert samples.mean(dim=0).numpy() == pytest.approx(np.ones(dim), abs=tol, rel=0)
    assert np.cov(samples, rowvar=False) == pytest.approx(
        cov.numpy() * temperature, abs=tol, rel=0
    )


def test_energy():
    """Test energy of the Normal Distribution."""

    dim = 2
    cov = torch.tensor([[1, 0.0], [0.0, 1]])
    mean = torch.ones(dim)

    normal_distribution = NormalDistribution(dim, mean=mean, cov=cov)
    energy = normal_distribution.energy(torch.tensor([1, 2]))
    assert energy.numpy() == pytest.approx(
        2 / 2 * np.log(2 * np.pi) + 0.5 * 1, abs=1e-3, rel=0
    )

    cov2 = torch.tensor([[1, 0.0], [0.0, 2]])
    normal_distribution2 = NormalDistribution(dim, mean=mean, cov=cov2)
    energy = normal_distribution2.energy(torch.tensor([1, 2]))
    assert energy.numpy() == pytest.approx(
        2 / 2 * np.log(2 * np.pi) + 0.5 * 1 / 2 + 0.5 * np.log(2), abs=1e-2, rel=0
    )


@pytest.mark.parametrize("assert_range", [True, False])
@pytest.mark.parametrize("sampling_method", ["icdf", "rejection"])
def test_truncated_normal_distribution_tensors(assert_range, sampling_method):
    """Test with tensor parameters."""
    dim = 5
    tn = TruncatedNormalDistribution(
        dim,
        mu=1 + torch.rand(dim,),
        sigma=torch.rand(dim,),
        lower_bound=torch.rand(dim,),
        upper_bound=4 * torch.ones(dim,),
        assert_range=assert_range,
        sampling_method=sampling_method,
    )
    n_samples = 10
    samples = tn.sample(n_samples)
    assert samples.shape == (n_samples, dim)
    assert torch.all(samples >= tn._lower_bound)
    assert torch.all(samples <= tn._upper_bound)
    energies = tn.energy(samples)
    assert energies.shape == (n_samples, 1)
    assert torch.all(torch.isfinite(energies))

    # failure test
    samples[torch.arange(0, 5), torch.arange(0, 5)] = -1.0
    if assert_range:
        with pytest.raises(ValueError):
            tn.energy(samples)
    else:
        energies = tn.energy(samples)
        assert torch.all(torch.isinf(energies[:5, 0]))
        assert torch.all(torch.isfinite(energies[5:, 0]))
