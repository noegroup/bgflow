import torch
import pytest
import numpy as np
from bgflow.distribution import NormalDistribution, TruncatedNormalDistribution, MeanFreeNormalDistribution
from bgflow.utils import as_numpy

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


def test_sample_energy_multi_temperature(ctx):
    dim = 1000
    torch.manual_seed(123445)
    temperature = torch.tensor([0.5, 1.0, 2], **ctx)[..., None]
    n_samples = len(temperature)
    mean = torch.ones(dim, **ctx)
    normal_distribution = NormalDistribution(dim, mean=mean, cov=torch.eye(dim, **ctx))

    samples = normal_distribution.sample(3, temperature=temperature)

    assert samples.shape == torch.Size([n_samples, dim])
    assert samples.mean().item() == pytest.approx(1.0, abs=5e-2, rel=0)
    assert as_numpy(samples.var(dim=1)) == pytest.approx(as_numpy(temperature.flatten()), abs=0.2, rel=0)

    x = torch.randn(1, 1000, **ctx).expand(3, 1000)
    energy = normal_distribution.energy(x, temperature=temperature)
    energy_t0 = energy[1]
    for i in [0, 2]:
        du = energy[i] - energy_t0 / temperature[i]
        du = as_numpy(du)
        assert du.std() < 1e-5


@pytest.mark.parametrize("sigma", [1, 8.0])
@pytest.mark.parametrize("temperature", [1.0, 0.5, 10.0])
def test_normalization_1d(ctx, sigma, temperature):
    if sigma == 1:
        # to check without the cov argument
        normal_1 = NormalDistribution(dim=1).to(**ctx)
    else:
        normal_1 = NormalDistribution(dim=1, cov=torch.tensor([[sigma**2]])).to(**ctx)
    normal_t = NormalDistribution(dim=1, cov=torch.tensor([[temperature*sigma**2]])).to(**ctx)
    nbins = 10000
    xmax = 3*sigma*np.sqrt(temperature)
    x = torch.linspace(-xmax, xmax, nbins, **ctx)[..., None]
    dx = 2 * xmax / nbins
    u1 = as_numpy(normal_1.energy(x))
    ut1 = as_numpy(normal_1.energy(x, temperature=temperature))
    ut = as_numpy(normal_t.energy(x))
    atol = 1e-4 if ctx["dtype"] is torch.float32 else 1e-5
    # check that the u_T = u_1 / T + const
    assert(u1 / temperature - ut).std() == pytest.approx(0.0, abs=atol)
    assert ut == pytest.approx(ut1, abs=atol)
    # check that the integral(e^-u) = 1
    assert (np.exp(-ut1) * dx).sum() == pytest.approx(1., abs=1e-2)


@pytest.mark.parametrize("temperature", [1.0, 0.5, 2.0, 31.41, torch.tensor([[1.0], [2.0]])])
def test_normalization_2d(ctx, temperature):
    """check the normalization constant at different temperatures"""
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(**ctx)
    dim = 2
    n_samples = 2
    cov = torch.tensor([[1, 0.3], [0.3, 2]], **ctx)
    mean = torch.ones(dim, **ctx)
    normal_distribution = NormalDistribution(dim, mean=mean, cov=cov)
    samples = normal_distribution.sample(n_samples, temperature=temperature)

    tt = torch.as_tensor(temperature)[..., None] if isinstance(temperature, torch.Tensor) else temperature
    ref = torch.distributions.MultivariateNormal(
        loc=mean, covariance_matrix=tt*cov
    )
    logp = as_numpy(ref.log_prob(samples))[..., None]
    u = as_numpy(normal_distribution.energy(samples, temperature=temperature))
    atol = 4e-3 if ctx["dtype"] == torch.float32 else 1e-5
    assert u == pytest.approx(-logp, abs=atol)
