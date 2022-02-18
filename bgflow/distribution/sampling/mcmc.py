"""Markov-Chain Monte Carlo iterative sampling.

The `MCMCStep` takes a target energy and a proposal.
Each proposal is a `torch.nn.Module`, whose `forward` method
takes a `SamplerState` and proposes a modified state along
with a `delta_log_prob` tensor. The `delta_log_prob` is the difference
in log proposal probabilities. It is 0 for symmetric proposals
`g(x|x') = g(x'|x)` and nonzero for asymmetric proposals,
`delta_log_prob = log g(x'|x) - log g(x|x')`.
"""

import warnings
from typing import Tuple

import torch

from ..energy import Energy
from .base import Sampler
from .iterative import SamplerStep, IterativeSampler, SamplerState
from ._iterative_helpers import default_set_samples_hook


__all__ = [
    "GaussianMCMCSampler", "MCMCStep", "GaussianProposal", "LatentProposal",
    "metropolis_accept"
]


class GaussianProposal(torch.nn.Module):
    """Normal distributed displacement of samples.

    Parameters
    ----------
    noise_std : float, optional
        The standard deviation by which samples are perturbed.
    """
    def __init__(self, noise_std=0.1):
        super().__init__()
        self._noise_std = noise_std

    def forward(self, state: SamplerState) -> Tuple[SamplerState, float]:
        delta_log_prob = 0.0  # symmetric density
        proposed_state = state.replace(
            samples=tuple(x + torch.randn_like(x) * self._noise_std for x in state.as_dict()["samples"])
        )
        return proposed_state, delta_log_prob


class LatentProposal(torch.nn.Module):
    """Proposal in latent space.

    Parameters
    ----------
    flow : bgflow.Flow
        A bijective transform, where the latent-to-target direction is the forward direction.
    base_proposal : torch.nn.Module, optional
        The proposal that is to be performed in latent space.
    flow_kwargs : dict, optional
        Any additional keyword arguments to the flow.
    """
    def __init__(
            self,
            flow,
            base_proposal=GaussianProposal(noise_std=0.1),
            flow_kwargs=dict()
    ):
        super().__init__()
        self.flow = flow
        self.base_proposal = base_proposal
        self.flow_kwargs = flow_kwargs

    def forward(self, state: SamplerState) -> Tuple[SamplerState, torch.Tensor]:
        *z, logdet_inverse = self.flow.forward(
            *state.as_dict()["samples"], inverse=True, **self.flow_kwargs
        )
        proposed_latent, delta_log_prob = self.base_proposal.forward(state.replace(samples=z))
        *proposed_samples, logdet_forward = self.flow.forward(*proposed_latent.as_dict()["samples"])
        # g(x|x') = g(x|z') = p_z (F_{zx}^{-1}(x)  | z') * log | det J_{zx}^{-1} (x) |
        # log g(x|x') = log p(z|z') + logabsdet J_{zx}^-1 (x)
        # log g(x'|x) = log p(z'|z) + logabsdet J_{zx}^-1 (x')
        # log g(x'|x) - log g(x|x') = log p(z|z') - log p(z|z') - logabsdet_forward - logsabdet_inverse
        delta_log_prob = delta_log_prob - (logdet_forward + logdet_inverse)
        return proposed_latent.replace(samples=proposed_samples), delta_log_prob[:, 0]


class MCMCStep(SamplerStep):
    """Metropolis Monte Carlo

    Parameters
    ----------
    target_energy : bgflow.Energy
    proposal : torch.nn.Module, optional
    target_temperatures : Union[torch.Tensor, float]
    n_steps : int
        Number of steps to take at a time.
    """
    def __init__(self, target_energy, proposal=GaussianProposal(), target_temperatures=1.0, n_steps=1):
        super().__init__(n_steps=n_steps)
        self.target_energy = target_energy
        self.target_temperatures = target_temperatures
        self.proposal = proposal

    def _step(self, state: SamplerState) -> SamplerState:
        # compute current energies
        state = state.evaluate_energy_force(self.target_energy, evaluate_forces=False)
        # make a proposal
        proposed_state, delta_log_prob = self.proposal.forward(state)
        proposed_state = proposed_state.evaluate_energy_force(self.target_energy, evaluate_forces=False)
        # accept according to Metropolis criterion
        new_dict = proposed_state.as_dict()
        old_dict = state.as_dict()
        accept = metropolis_accept(
            current_energies=old_dict["energies"]/self.target_temperatures,
            proposed_energies=new_dict["energies"]/self.target_temperatures,
            proposal_delta_log_prob=delta_log_prob
        )
        return state.replace(
            samples=tuple(
                torch.where(accept[..., None], new, old) for new, old in zip(new_dict["samples"], old_dict["samples"])
            ),
            energies=torch.where(accept, new_dict["energies"], old_dict["energies"])
        )


class GaussianMCMCSampler(IterativeSampler):
    """This is a shortcut implementation of a simple Gaussian MCMC sampler
    that is largely backward-compatible with the old implementation.
    The only difference is that `GaussianMCMCSampler.sample(n)`
    will propagate for `n` strides rather than `n` steps.

    Parameters
    ----------
    energy : bgflow.Energy
        The target energy.
    init_state : Union[torch.Tensor, SamplerState]
    temperature : Union[torch.Tensor, float], optional
        The temperature scaling factor that is broadcasted along the batch dimension.
    noise_std : float, optional
        The Gaussian noise standard deviation.
    stride : int, optional
    n_burnin : int, optional
    box_constraint : Callable, optional
        The function is supplied as a `set_samples_hook` to the SamplerState so that
        boundary conditions are applied to all samples.
    return_hook : Callable, optional
        The function is supplied as a `return_hook` to the Sampler. By default, we combine
        the batch and sample dimensions to keep consistent with the old implementation.
    """
    def __init__(
            self,
            energy,
            init_state,
            temperature=1.,
            noise_std=.1,
            stride=1,
            n_burnin=0,
            box_constraint=None,
            return_hook=None,
            **kwargs
    ):
        # first, some things to ensure backwards compatibility
        # apply the box constraint function whenever samples are set
        set_samples_hook = default_set_samples_hook
        if box_constraint is not None:
            set_samples_hook = lambda samples: [box_constraint(x) for x in samples]
        if not isinstance(init_state, SamplerState):
            init_state = SamplerState(samples=init_state, set_samples_hook=set_samples_hook)
        # flatten batches before returning
        if return_hook is None:
            return_hook = lambda samples: [
                x.reshape(-1, *shape) for x, shape in zip(samples, energy.event_shapes)
            ]
        if "n_stride" in kwargs:
            warnings.warn("keyword n_stride is deprecated, use stride instead", DeprecationWarning)
            stride = kwargs["n_stride"]
        # set up sampler
        super().__init__(
            init_state,
            sampler_steps=[
                MCMCStep(
                    energy,
                    proposal=GaussianProposal(noise_std=noise_std),
                    target_temperatures=temperature,
                ),
            ],
            stride=stride,
            n_burnin=n_burnin,
            return_hook=return_hook
        )


def metropolis_accept(
        current_energies,
        proposed_energies,
        proposal_delta_log_prob
):
    """Metropolis criterion.

    Parameters
    ----------
    current_energies : torch.Tensor
        Dimensionless energy of the current state x.
    proposed_energies : torch.Tensor
        Dimensionless energy of the proposed state x'.
    proposal_delta_log_prob : Union[torch.Tensor, float]
        The difference in log probabilities between the forward and backward proposal.
        This corresponds to    log g(x'|x) - log g(x|x'), where g is the proposal distribution.

    Returns
    -------
    accept : torch.Tensor
        A boolean tensor that is True for accepted proposals and False otherwise
    """
    # log p(x') - log p(x) - (log g(x'|x) - log g(x|x'))
    log_prob = -(proposed_energies - current_energies) - proposal_delta_log_prob
    log_acceptance_ratio = torch.min(
        torch.zeros_like(proposed_energies),
        log_prob,
    )
    log_random = torch.rand_like(log_acceptance_ratio).log()
    accept = log_acceptance_ratio >= log_random
    return accept


class _GaussianMCMCSampler(Energy, Sampler):
    """Deprecated legacy implementation."""
    def __init__(
            self,
            energy,
            init_state=None,
            temperature=1.,
            noise_std=.1,
            n_stride=1,
            n_burnin=0,
            box_constraint=None
    ):
        super().__init__(energy.dim)
        warnings.warn(
            """This implementation of the MC sampler is deprecated. 
Instead try using:
>>> IterativeSampler(
>>>     init_state, [MCMCStep(energy)]
>>> ) 
""",
            DeprecationWarning
        )
        self._energy_function = energy
        self._init_state = init_state
        self._temperature = temperature
        self._noise_std = noise_std
        self._n_stride = n_stride
        self._n_burnin = n_burnin
        self._box_constraint = box_constraint

        self._reset(init_state)

    def _step(self):
        noise = self._noise_std * torch.Tensor(self._x_curr.shape).normal_()
        x_prop = self._x_curr + noise
        e_prop = self._energy_function.energy(x_prop, temperature=self._temperature)
        e_diff = e_prop - self._e_curr
        r = -torch.Tensor(x_prop.shape[0]).uniform_(0, 1).log().view(-1, 1)
        acc = (r > e_diff).float().view(-1, 1)
        rej = 1. - acc
        self._x_curr = rej * self._x_curr + acc * x_prop
        self._e_curr = rej * self._e_curr + acc * e_prop
        if self._box_constraint is not None:
            self._x_curr = self._box_constraint(self._x_curr)
        self._xs.append(self._x_curr)
        self._es.append(self._e_curr)
        self._acc.append(acc.bool())

    def _reset(self, init_state):
        self._x_curr = self._init_state
        self._e_curr = self._energy_function.energy(self._x_curr, temperature=self._temperature)
        self._xs = [self._x_curr]
        self._es = [self._e_curr]
        self._acc = [torch.zeros(init_state.shape[0]).bool()]
        self._run(self._n_burnin)

    def _run(self, n_steps):
        with torch.no_grad():
            for i in range(n_steps):
                self._step()

    def _sample(self, n_samples):
        self._run(n_samples)
        return torch.cat(self._xs[-n_samples::self._n_stride], dim=0)

    def _sample_accepted(self, n_samples):
        samples = self._sample(n_samples)
        acc = torch.cat(self._acc[-n_samples::self._n_stride], dim=0)
        return samples[acc]

    def _energy(self, x):
        return self._energy_function.energy(x)

