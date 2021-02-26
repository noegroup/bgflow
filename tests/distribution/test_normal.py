
import torch
from bgtorch.distribution.normal import TruncatedNormalDistribution


def test_truncated_normal_distribution_tensors():
    """Test with tensor parameters."""
    dim = 5
    tn = TruncatedNormalDistribution(
        dim,
        mu=1+torch.rand(dim,),
        sigma=torch.rand(dim,),
        lower_bound=torch.rand(dim,),
        upper_bound=4*torch.ones(dim,)
    )
    n_samples = 10
    samples = tn.sample(n_samples)
    assert samples.shape == (n_samples, dim)
    assert torch.all(samples >= tn._lower_bound)
    assert torch.all(samples <= tn._upper_bound)
    energies = tn.energy(samples)
    assert energies.shape == (n_samples, 1)
    samples[torch.arange(0,5),torch.arange(0,5)] = -1.0
    energies = tn.energy(samples)
    assert torch.all(torch.isinf(energies[:5,0]))
    assert torch.all(torch.isfinite(energies[5:,0]))
