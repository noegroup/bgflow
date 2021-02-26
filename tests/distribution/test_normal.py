
import pytest

import torch
from bgtorch.distribution.normal import TruncatedNormalDistribution


@pytest.mark.parametrize("assert_range", [True, False])
def test_truncated_normal_distribution_tensors(assert_range):
    """Test with tensor parameters."""
    dim = 5
    tn = TruncatedNormalDistribution(
        dim,
        mu=1+torch.rand(dim,),
        sigma=torch.rand(dim,),
        lower_bound=torch.rand(dim,),
        upper_bound=4*torch.ones(dim,),
        assert_range=assert_range
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
        assert torch.all(torch.isinf(energies[:5,0]))
        assert torch.all(torch.isfinite(energies[5:,0]))

