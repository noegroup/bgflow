"""Iterative sampler framework.
The framework is designed in two layers.

1. A driver class `IterativeSampler` that has an internal state `SamplerState`
2. Worker classes based on `SamplerStep`

The sampler steps (subclasses of `SamplerStep`) implement a `_step` method
that receives a `SamplerState` and returns a modified `SamplerState.`
Those states contain a minibatch of samples and (optionally) metainformation such
as corresponding velocities or periodic box dimensions.

Each sample in the minibatch can be propagated according to a different temperature.

Examples
--------

An MCMC sampler is set up as follows
>>> from bgflow import DoubleWellEnergy, IterativeSampler, SamplerState, MCMCStep
>>> energy = DoubleWellEnergy(dim=10)
>>> sampler_state = SamplerState(samples=torch.randn(3, 10))
>>> mcmc_step = MCMCStep(energy, target_temperatures=torch.tensor([1., 10., 100.]))
>>> sampler = IterativeSampler(sampler_state, sampler_steps=[mcmc_step], stride=10, n_burnin=100)
>>> sampler.sample(10)

"""

import torch
from .base import Sampler
from ...utils.types import pack_tensor_in_list

__all__ = ["SamplerState", "default_extract_sample_hook", "IterativeSampler", "SamplerStep"]


def default_set_samples_hook(x):
    """by default, use samples as is"""
    return x


class SamplerState(dict):
    """State of an iterative sampler.
    Contains a minibatch of samples alongside optional information such as
    velocities, forces.

    The SamplerState is essentially a dictionary that can contain any additional
    user-defined fields.

    >>> state = SamplerState(samples=torch.randn(10, 10), some_other_variable=torch.randn(10, 10))

    The sampler steps that operate on a sampler state can read from and write to
    these user-defiend variables.
    Fields can be accessed by the syntax

    >>> state.samples, state.some_other_variable

    which is equivalent to

    >>> state["samples"], state["some_other_variable"]

    Notes
    -----
    To support energies that are defined on multiple events, the samples are
    stored internally as a list of tensors.
    (In most cases, this list will only contain one tensor.)

    Parameters
    ----------
    samples : Union[torch.Tensor, list[torch.Tensor]]
    energies : Union[torch.Tensor, NoneType], optional
    velocities : Union[torch.Tensor, list[torch.Tensor], NoneType], optional
    forces : Union[torch.Tensor, list[torch.Tensor], NoneType], optional
    box_vectors : Union[torch.Tensor, NoneType], optional
        The box vectors are a 3x3 matrix (a b c) , whose columns denote the periodic box vectors.
        The first vector, a, has to be parallel to the x-axis. The second vector, b, has
        to lie in the x-y plane (i.e. the matrix is upper triangular).
    set_samples_hook : Callable, optional
        A callable that is applied to the samples whenever samples are set.
    _are_energies_up_to_date : bool, optional
    _are_forces_up_to_date : bool, optional

    """
    def __init__(
            self,
            samples,
            energies=None,
            velocities=None,
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
            velocities=pack_tensor_in_list(velocities),
            forces=pack_tensor_in_list(forces),
            box_vectors=box_vectors,
            **kwargs
        )
        self._are_energies_up_to_date = _are_energies_up_to_date
        self._are_forces_up_to_date = _are_forces_up_to_date
        self.set_samples_hook = set_samples_hook

    def __setitem__(self, key, value):
        if key == "samples":
            value = self.set_samples_hook(pack_tensor_in_list(value))
        super().__setitem__(key, value)
        if key in ["samples", "box_vectors"]:
            # invalidate energies and forces
            self._are_energies_up_to_date = False
            self._are_forces_up_to_date = False
        elif key == "forces":
            self._are_forces_up_to_date = True
        elif key == "energies":
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
        """Make a shallow copy.

        Returns
        -------
        copy : SamplerState
        """
        clone = SamplerState(**self)
        for item, value in clone.items():
            if isinstance(value, list):
                clone[item] = value.copy()
        return clone

    def needs_update(self, check_energies=True, check_forces=False):
        """Whether the energies and forces are outdated.
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
    sampler_state : SamplerState
        The state of the sampler. The sampler state has an attribute `samples`,
        which contains a minibatch of samples.
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

    Notes
    -----
    The class implements the interface of an iterable-style torch dataset
    and can thus be used within a torch DataLoader.
    """
    def __init__(
            self,
            sampler_state,
            sampler_steps,
            stride=1,
            n_burnin=0,
            max_iterations=None,
            extract_sample_hook=default_extract_sample_hook,
            progress_bar=lambda x: x,
            **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(sampler_state, torch.Tensor):
            sampler_state = SamplerState(samples=sampler_state)
        self.state = sampler_state
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
                samples = [x[None, ...].clone() for x in new_samples]
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
    """Abstract base class for iterative sampler steps.
    A SamplerStep implements a `_step` function,
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


