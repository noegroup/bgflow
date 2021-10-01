
import torch
from bgflow import IterativeSampler, SamplerState
from bgflow.utils.types import pack_tensor_in_list
from bgflow.distribution.sampling.md import OpenMMStep
from ..energy.test_openmm import PeriodicTwoParticleTestBridge


def test_openmm_sampling(ctx):
    bridge = PeriodicTwoParticleTestBridge(1, n_simulation_steps=10)

    batchsize = 4
    sampler_state = SamplerState(
        samples=torch.randn(batchsize, 6, **ctx),
        velocities=pack_tensor_in_list(torch.randn(batchsize, 6, **ctx)),
        box_vectors=torch.arange(2, batchsize+2)[:, None, None] * torch.eye(3, **ctx)
    )

    sampler = IterativeSampler(
        sampler_state=sampler_state,
        sampler_steps=[OpenMMStep(bridge)]
    )

    assert sampler.sample(10).shape == (10, batchsize, 6)

    sampler.return_hook = lambda x: x[0].reshape(-1, 6)
    assert sampler.sample(10).shape == (10*batchsize, 6)
