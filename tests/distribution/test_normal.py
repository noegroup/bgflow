import torch
import pytest
import numpy as np
from bgflow.distribution import NormalDistribution, TruncatedNormalDistribution, MeanFreeNormalDistribution

@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize("temperature", [1.0, 0.5, 2])
def test_normal_distribution(device, dtype, dim, n_samples, temperature):
    """Test sampling of the Normal Distribution."""

    cov = torch.tensor([[1, 0.3], [0.3, 2]], device=device, dtype=dtype)
    mean = torch.ones(dim)
    normal_distribution = NormalDistribution(dim, mean=mean, cov=cov)

    samples = normal_distribution.sample(n_samples, temperature=temperature)
    tol = 0.2 * np.sqrt(temperature)
    assert samples.shape == torch.Size([n_samples, dim])
    assert samples.mean(dim=0).cpu().numpy() == pytest.approx(np.ones(dim), abs=tol, rel=0)
    assert np.cov(samples.cpu().numpy(), rowvar=False) == pytest.approx(
        cov.cpu().numpy() * temperature, abs=tol, rel=0
    )


def test_energy(device, dtype):
    """Test energy of the Normal Distribution."""

    dim = 2
    cov = torch.tensor([[1, 0.0], [0.0, 1]], device=device, dtype=dtype)
    mean = torch.ones(dim).to(cov)

    normal_distribution = NormalDistribution(dim, mean=mean, cov=cov)
    energy = normal_distribution.energy(torch.tensor([1, 2]).to(mean))
    assert energy.cpu().numpy() == pytest.approx(
        2 / 2 * np.log(2 * np.pi) + 0.5 * 1, abs=1e-3, rel=0
    )

    cov2 = torch.tensor([[1, 0.0], [0.0, 2]]).to(cov)
    normal_distribution2 = NormalDistribution(dim, mean=mean, cov=cov2)
    energy = normal_distribution2.energy(torch.tensor([1, 2]).to(cov))
    assert energy.cpu().numpy() == pytest.approx(
        2 / 2 * np.log(2 * np.pi) + 0.5 * 1 / 2 + 0.5 * np.log(2), abs=1e-2, rel=0
    )


@pytest.mark.parametrize("assert_range", [True, False])
@pytest.mark.parametrize("sampling_method", ["icdf", "rejection"])
@pytest.mark.parametrize("is_learnable", [True, False])
def test_truncated_normal_distribution_tensors(device, dtype, assert_range, sampling_method, is_learnable):
    """Test with tensor parameters."""
    dim = 5
    tn = TruncatedNormalDistribution(
        mu=1 + torch.rand(dim, device=device, dtype=dtype),
        sigma=torch.rand(dim,),
        lower_bound=torch.rand(dim,),
        upper_bound=4 * torch.ones(dim,),
        assert_range=assert_range,
        sampling_method=sampling_method,
        is_learnable=is_learnable
    ).to(device, dtype)
    if is_learnable:
        assert len(list(tn.parameters())) > 0
    else:
        assert len(list(tn.parameters())) == 0
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


@pytest.mark.parametrize("two_event_dims", [True, False])
def test_mean_free_normal_distribution(two_event_dims, device):
    mean_free_normal_distribution = MeanFreeNormalDistribution(dim=8, n_particles=4, two_event_dims=two_event_dims)
    mean_free_normal_distribution.to(device)
    n_samples = 10000
    threshold = 1e-5
    samples = mean_free_normal_distribution.sample(n_samples)
    if two_event_dims:
        assert samples.shape == (n_samples, 4, 2)
        mean_deviation = samples.mean(dim=(1, 2))
        assert torch.all(mean_deviation.abs() < threshold)
    else:
        assert samples.shape == (n_samples, 8)
        mean_deviation = samples.mean(dim=(1))
        assert torch.all(mean_deviation.abs() < threshold)