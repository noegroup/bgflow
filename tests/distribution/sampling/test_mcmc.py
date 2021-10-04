
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
        flow=BentIdentity(),  #SequentialFlow([BentIdentity()]*2),
        base_proposal=GaussianProposal(noise_std=0.4))
])
@pytest.mark.parametrize("temperatures", [torch.ones(3), torch.arange(1, 10, 100)])
def test_mcmc(ctx, proposal, temperatures):
    """sample from a normal distribution with mu=3 and std=1,2,3 using MCMC"""
    try:
        import tqdm
        progress_bar = tqdm.tqdm
    except ImportError:
        progress_bar = None
    target = NormalDistribution(4, 3.0*torch.ones(4, **ctx))
    temperatures = temperatures.to(**ctx)
    # for testing efficiency we have a second batch dimension
    # this is not required; we could remove the first batch dimension (256)
    # and just sample longer
    state = SamplerState(samples=0.0*torch.ones(512, 3, 4, **ctx))
    mcmc = IterativeSampler(
        initial_state=state,
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
    samples = mcmc.sample(100)  # 200 -> 2.72, 2.73
    assert torch.allclose(samples.mean(dim=(0, 1, 3)), torch.tensor([3.0]*3, **ctx), atol=0.1)
    std = samples.std(dim=(0, 1, 3))
    assert torch.allclose(std, temperatures.sqrt(), rtol=0.05, atol=0.0)


def test_old_vs_new_mcmc(ctx):
    energy = NormalDistribution(dim=4)
    x0 = torch.randn(64, 4)
    old_mc = _GaussianMCMCSampler(energy, x0)
    new_mc = GaussianMCMCSampler(energy, x0)
    assert old_mc.sample(10).shape == new_mc.sample(10).shape
