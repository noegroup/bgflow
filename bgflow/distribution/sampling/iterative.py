
from collections import namedtuple
from types import SimpleNamespace
import torch
from .base import Sampler
from ...utils.types import unpack_tensor_tuple, pack_tensor_in_list

__all__ = ["SamplerState", "default_get_sample_hook", "IterativeSampler", "SamplerStep"]


class SamplerState(dict):
    """Batch of states of iterative samplers.
    Contains samples, energies (optional), momenta (optional), forces (optional)
    along an arbitrary number of batch dimensions.
    """
    def __init__(self, samples, energies=None, momenta=None, forces=None, box_vectors=None, **kwargs):
        super().__init__(
            samples=pack_tensor_in_list(samples),
            energies=energies,
            momenta=pack_tensor_in_list(momenta),
            forces=pack_tensor_in_list(forces),
            box_vectors=box_vectors,
            **kwargs
        )

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, item, value):
        return self.__setitem__(item, value)


def default_get_sample_hook(state: SamplerState):
    return state.samples


class IterativeSampler(Sampler, torch.utils.data.Dataset):
    """The iterative sampler manages the sampling from a target energy
    using iterative methods (MCMC, ...).

    Parameters
    ----------
    initial_state : SamplerState
        The initial state for the sampler. The sampler state has an attribute `Samples`
        which
    sampler_steps : Sequence[SamplerStep]
        A list of sampler_steps, which are applied
    """
    def __init__(
            self,
            initial_state,
            sampler_steps,
            get_sample_hook=default_get_sample_hook,
            progress_bar=lambda x: x,
            stride=1,
            n_burnin=0,
            max_iterations=None
    ):
        super().__init__()
        self.state = initial_state
        self.sampler_steps = sampler_steps
        self.get_sample_hook = get_sample_hook
        self.progress_bar = progress_bar
        self.stride = stride
        self.max_iterations = max_iterations
        self.i = 0
        for _ in self.progress_bar(range(n_burnin)):
            self.state = next(self)

    def _sample_with_temperature(self, n_samples, temperature):
        raise NotImplementedError()

    def _sample(self, n_samples):
        samples = None
        for _ in self.progress_bar(range(n_samples)):
            self.state = next(self)
            new_samples = self.get_sample_hook(self.state)
            if samples is None:
                # add batch dim
                samples = [x[None, ...] for x in new_samples]
            else:
                for i, (x, new) in enumerate(zip(samples, new_samples)):
                    samples[i] = torch.cat((x, new[None, ...]), dim=0)
        return samples

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_iterations is not None and self.i >= self.max_iterations:
            raise StopIteration
        for _ in range(self.stride):
            for sampler_step in self.sampler_steps:
                self.state = sampler_step.forward(self.state)
        self.i += 1
        return self.state


class SamplerStep(torch.nn.Module):
    """A SamplerStep implements a `_step` function,
    which receives a sampler state
    and returns a modified sampler state.

    Parameters
    ----------
    target_energies: list or torch.nn.Module
        The energies to sample from.

    target_temperatures: torch.tensor
        The temperatures to sample from.

    n_steps: int
        The number of steps taken per `forward` call.
    """
    def __init__(self, n_steps=1):
        super().__init__()
        self._n_steps = n_steps

    def _step(self, state):
        raise NotImplementedError()

    def forward(self, state):
        for _ in range(self._n_steps):
            state = self._step(state)
        return state


