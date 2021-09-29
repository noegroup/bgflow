
import torch
from .base import Sampler
from ...utils.types import pack_tensor_in_list

__all__ = ["SamplerState", "default_get_sample_hook", "IterativeSampler", "SamplerStep"]


class SamplerState(dict):
    """Batch of states of iterative samplers.
    Contains samples, energies (optional), momenta (optional), forces (optional)
    along an arbitrary number of batch dimensions.
    """
    def __init__(
            self,
            samples,
            energies=None,
            momenta=None,
            forces=None,
            box_vector_min=None,
            box_vector_max=None,
            **kwargs
    ):
        super().__init__(
            samples=pack_tensor_in_list(samples),
            energies=energies,
            momenta=pack_tensor_in_list(momenta),
            forces=pack_tensor_in_list(forces),
            box_vectors=[box_vector_min, box_vector_max],
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
    get_sample_hook : Callable, optional
        A function that takes a sampler state and returns the samples
    progress_bar : Callable, optional
        A progress bar (e.g. tqdm.tqdm)
    stride : int, optional
        The number of steps to take between two samples.
    n_burnin : int, optional
        Number of steps to be taken before starting to sample.
    max_iterations : int,optional
        The maximum number of steps this sampler can take. None = infinitely many.
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
        if isinstance(initial_state, torch.Tensor):
            initial_state = SamplerState(samples=initial_state)
        self.state = initial_state
        self.sampler_steps = sampler_steps
        self.get_sample_hook = get_sample_hook
        self.progress_bar = progress_bar
        self.stride = stride
        self.max_iterations = max_iterations
        self.i = 0
        for _ in self.progress_bar(range(n_burnin)):
            self.state = next(self)

    def _sample(self, n_samples, *args, **kwargs):
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


