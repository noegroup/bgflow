
import pytest
import torch
from bgflow import (
    GaussianProposal, SamplerState, NormalDistribution,
    IterativeSampler, MCMCStep, LatentProposal, SequentialFlow, BentIdentity
)


@pytest.mark.parametrize("proposal", [
    GaussianProposal(noise_std=0.2),
    LatentProposal(
        flow=SequentialFlow([BentIdentity()]*2),
        base_proposal=GaussianProposal(noise_std=0.2))
])
@pytest.mark.parametrize("temperatures", [torch.ones(3), torch.arange(1, 4)])
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
        sampler_state=state,
        sampler_steps=[
            MCMCStep(
                target,
                proposal=proposal.to(**ctx),
                target_temperatures=temperatures
            )
        ],
        stride=5,
        n_burnin=100,
        progress_bar=progress_bar
    )
    samples = mcmc.sample(200)
    assert samples.mean().item() == pytest.approx(3.0, abs=0.1)
    std = samples.std(dim=(0, 1, 3))
    assert torch.allclose(std, temperatures.sqrt(), atol=0.2)


