

import torch
from bgflow import IterativeSampler, SamplerState, SamplerStep
from bgflow.distribution.sampling._iterative_helpers import AbstractSamplerState


class AddOne(SamplerStep):
    def _step(self, state: AbstractSamplerState):
        statedict = state.as_dict()
        samples = tuple(
            x + 1.0
            for x in statedict["samples"]
        )
        return state.replace(samples=samples)


def test_iterative_sampler(ctx):
    state = SamplerState(samples=[torch.zeros(2, **ctx), ])

    # test burnin
    sampler = IterativeSampler(state, sampler_steps=[AddOne()], n_burnin=10)
    assert torch.allclose(sampler.state.samples[0], 10*torch.ones_like(sampler.state.samples[0]))

    # test sampling
    samples = sampler.sample(2)
    assert torch.allclose(samples, torch.tensor([[11., 11.], [12., 12.]], **ctx))

    # test stride
    sampler.stride = 5
    samples = sampler.sample(2)
    assert sampler.i == 14
    assert torch.allclose(samples, torch.tensor([[17., 17.], [22., 22.]], **ctx))

    # test iteration
    sampler.max_iterations = 15
    for batch in sampler:  # only called once
        assert torch.allclose(batch.samples[0], torch.tensor([[27., 27.]], **ctx))
    sampler.max_iterations = None

