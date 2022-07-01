import warnings

import pytest
import torch
from bgflow import (
    GaussianProposal, SamplerState, NormalDistribution,
    IterativeSampler, MCMCStep, LatentProposal, BentIdentity, GaussianMCMCSampler
)
from bgflow.distribution.sampling.mcmc import _GaussianMCMCSampler


@pytest.mark.parametrize("proposal", [
    GaussianProposal(noise_std=0.2),
    LatentProposal(
        flow=BentIdentity(),
        base_proposal=GaussianProposal(noise_std=0.4))
])
@pytest.mark.parametrize("temperatures", [torch.ones(3), torch.arange(1, 10, 100)])
def test_mcmc(ctx, proposal, temperatures):
    """sample from a normal distribution with mu=3 and std=1,2,3 using MCMC"""
    try:
        import tqdm
        progress_bar = tqdm.tqdm
    except ImportError:
        progress_bar = lambda x: x
    target = NormalDistribution(4, 3.0*torch.ones(4, **ctx))
    temperatures = temperatures.to(**ctx)
    # for testing efficiency we have a second batch dimension
    # this is not required; we could remove the first batch dimension (256)
    # and just sample longer
    state = SamplerState(samples=0.0*torch.ones(512, 3, 4, **ctx))
    mcmc = IterativeSampler(
        sampler_state=state,
        sampler_steps=[
            MCMCStep(
                target,
                proposal=proposal.to(**ctx),
                target_temperatures=temperatures
            )
        ],
        stride=2,
        n_burnin=100,
        progress_bar=progress_bar
    )
    samples = mcmc.sample(100)
    assert torch.allclose(samples.mean(dim=(0, 1, 3)), torch.tensor([3.0]*3, **ctx), atol=0.1)
    std = samples.std(dim=(0, 1, 3))
    assert torch.allclose(std, temperatures.sqrt(), rtol=0.05, atol=0.0)


def test_old_vs_new_mcmc(ctx):
    energy = NormalDistribution(dim=4)
    x0 = torch.randn(64, 4)

    def constraint(x):
        return torch.fmod(x,torch.ones(4))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_mc = _GaussianMCMCSampler(
            energy, x0, n_stride=10, n_burnin=10,
            noise_std=0.3, box_constraint=constraint
        )
    new_mc = GaussianMCMCSampler(
        energy, x0, stride=10, n_burnin=10,
        noise_std=0.3, box_constraint=constraint
    )
    old_samples = old_mc.sample(1000)
    new_samples = new_mc.sample(100)
    assert old_samples.shape == (6400, 4)
    assert old_samples.shape == new_samples.shape
    assert old_samples.mean().item() == pytest.approx(new_samples.mean().item(), abs=1e-2)
    assert old_samples.std().item() == pytest.approx(new_samples.std().item(), abs=1e-1)
