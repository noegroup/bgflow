import copy

import torch
from .base import Sampler
from ...utils.types import pack_tensor_in_list

__all__ = ["SamplerState", "default_extract_sample_hook", "IterativeSampler", "SamplerStep"]


def default_set_samples_hook(x):
    return x


class SamplerState(dict):
    """Batch of states of iterative samplers.
    Contains samples, energies (optional), velocities (optional), forces (optional)
    along an arbitrary number of batch dimensions.
    """
    def __init__(
            self,
            samples,
            energies=None,
            momenta=None,
            forces=None,
            box_vectors=None,
            set_samples_hook=default_set_samples_hook,
            _are_energies_up_to_date=False,
            _are_forces_up_to_date=False,
            **kwargs
    ):
        super().__init__(
            samples=pack_tensor_in_list(samples),
            energies=energies,
            momenta=pack_tensor_in_list(momenta),
            forces=pack_tensor_in_list(forces),
            box_vectors=pack_tensor_in_list(box_vectors),
            **kwargs
        )
        self._are_energies_up_to_date = _are_energies_up_to_date
        self._are_forces_up_to_date = _are_forces_up_to_date
        self.set_samples_hook = set_samples_hook

    @property
    def samples(self):
        return self["samples"]

    @samples.setter
    def samples(self, samples):
        self["samples"] = self.set_samples_hook(pack_tensor_in_list(samples))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in ["samples", "box_vectors"]:
            # invalidate energies and forces
            self._are_energies_up_to_date = False
            self._are_forces_up_to_date = False
        elif key == ["forces"]:
            self._are_forces_up_to_date = True
        elif key == ["energies"]:
            self._are_energies_up_to_date = True

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, item, value):
        return self.__setitem__(item, value)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self._are_energies_up_to_date == other._are_energies_up_to_date
            and self._are_forces_up_to_date == other._are_forces_up_to_date
            and self.set_samples_hook == other.set_samples_hook
        )

    def copy(self):
        return SamplerState(**self)

    def needs_update(self, check_energies=True, check_forces=False):
        """whether the energies and forces are outdated.
        The criteria is that energies/forces were set after samples were set.
        """
        if check_energies and not self._are_energies_up_to_date:
            return True
        elif check_energies and self["energies"] is None:
            return True
        elif check_forces and not self._are_forces_up_to_date:
            return True
        elif check_forces and self["forces"] is None:
            return True
        else:
            return False


def default_extract_sample_hook(state: SamplerState):
    """Default extraction of samples from a SamplerState."""
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
    stride : int, optional
        The number of steps to take between two samples.
    n_burnin : int, optional
        Burnin phase; number of samples for equilibration before starting to return samples.
    max_iterations : int,optional
        The maximum number of steps this sampler can take. None = infinitely many.
    extract_sample_hook : Callable, optional
        A function that takes a sampler state and returns the samples
    progress_bar : Callable, optional
        A progress bar (e.g. tqdm.tqdm)
    """
    def __init__(
            self,
            initial_state,
            sampler_steps,
            stride=1,
            n_burnin=0,
            max_iterations=None,
            extract_sample_hook=default_extract_sample_hook,
            progress_bar=lambda x: x,
            **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(initial_state, torch.Tensor):
            initial_state = SamplerState(samples=initial_state)
        self.state = initial_state
        self.sampler_steps = sampler_steps
        self.extract_sample_hook = extract_sample_hook
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
            new_samples = self.extract_sample_hook(self.state)
            if samples is None:
                # add batch dim
                samples = [copy.deepcopy(x[None, ...]) for x in new_samples]
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


