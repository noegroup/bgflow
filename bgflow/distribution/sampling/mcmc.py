"""Markov-Chain Monte Carlo iterative sampling."""

import warnings

import torch

from ..energy import Energy
from .base import Sampler
from .iterative import SamplerStep, IterativeSampler, SamplerState


__all__ = [
    "GaussianMCMCSampler", "MCMCStep", "GaussianProposal", "LatentProposal",
    "metropolis_accept"
]


class GaussianProposal(torch.nn.Module):
    def __init__(self, noise_std=0.1):
        super().__init__()
        self._noise_std = noise_std

    def forward(self, state):
        proposed_state = state.copy()  # shallow copy
        proposed_state.samples = [x + torch.randn_like(x) * self._noise_std for x in state.samples]
        delta_log_prob = 0.0  # symmetric density
        return proposed_state, delta_log_prob


class LatentProposal(torch.nn.Module):
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

    def forward(self, state):
        proposed_state = state.copy()  # shallow copy
        *proposed_state.samples, logdet_inverse = self.flow.forward(
            *state.samples, inverse=True, **self.flow_kwargs
        )
        proposed_state, delta_log_prob = self.base_proposal.forward(proposed_state)
        *proposed_state.samples, logdet_forward = self.flow.forward(*proposed_state.samples)
        # g(x|x') = g(x|z') = p_z (F_{zx}^{-1}(x)  | z') * log | det J_{zx}^{-1} (x) |
        # log g(x|x') = log p(z|z') + logabsdet J_{zx}^-1 (x)
        # log g(x'|x) = log p(z'|z) + logabsdet J_{zx}^-1 (x')
        # log g(x'|x) - log g(x|x') = log p(z|z') - log p(z|z') - logabsdet_forward - logsabdet_inverse
        delta_log_prob = delta_log_prob - (logdet_forward + logdet_inverse)
        return proposed_state, delta_log_prob[:, 0]


class MCMCStep(SamplerStep):
    """Metropolis Monte Carlo"""
    def __init__(self, target_energy, proposal=GaussianProposal(), target_temperatures=1.0, n_steps=1):
        super().__init__(n_steps=n_steps)
        self.target_energy = target_energy
        self.target_temperatures = target_temperatures
        self.proposal = proposal

    def _step(self, state: SamplerState):
        # compute current energies
        if state.needs_update(check_energies=True, check_forces=False):
            state.energies = self.target_energy.energy(*state.samples)[..., 0]
        # make a proposal
        proposed_state, delta_log_prob = self.proposal.forward(state)
        proposed_energies = self.target_energy.energy(*proposed_state.samples)[..., 0]
        # accept according to Metropolis criterion
        accept = metropolis_accept(
            current_energies=state.energies/self.target_temperatures,
            proposed_energies=proposed_energies/self.target_temperatures,
            proposal_delta_log_prob=delta_log_prob
        )
        state.samples = [
            torch.where(accept[..., None], new, old)
            for new, old in zip(proposed_state.samples, state.samples)
        ]
        state.energies = torch.where(accept, proposed_energies, state.energies)
        return state


class GaussianMCMCSampler(IterativeSampler):
    def __init__(
            self,
            energy,
            init_state=None,
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
        set_samples_hook = lambda x: x
        if box_constraint is not None:
            set_samples_hook = lambda samples: [box_constraint(x) for x in samples]
        init_state = init_state if isinstance(init_state, SamplerState) else SamplerState(init_state)
        init_state.set_samples_hook = set_samples_hook
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


class ReplicaExchangeStep(SamplerStep):
    def __init__(self, energy, temperature_scalings):
        super().__init__()
        # TODO
        self.energy = energy
        self.temperature_scalings = temperature_scalings
        self.is_odd_step = False

    def _step(self, state):
        if state.needs_update(check_energies=True, check_forces=False):
            state.energies = self.energy(*state.samples)[..., None]
        replica_index_lower = torch.arange(self.is_odd_step, state.energies.shape[-1] - 1, 2)
        replica_index_higher = replica_index_lower + 1
        lower_energies = state.energies[..., replica_index_higher]
        higher_energies = state.energies[..., replica_index_higher]
        lower_temperatures = self.temperature_scalings[replica_index_lower]
        higher_temperatures = self.temperature_scalings[replica_index_higher]
        accept = metropolis_accept(
            current_energies=lower_energies/lower_temperatures + higher_energies/higher_temperatures,
            proposed_energies=lower_energies/higher_temperatures + higher_energies/lower_temperatures,
            proposal_delta_log_prob=0.0
        )
        ...
        # perform swaps
        # TODO (Michele)
        # rescale velocities
        if state.velocities is not None:
            # TODO
            pass
        return state


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
    """Deprecated implementation."""
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
        shape = self._x_curr.shape
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

