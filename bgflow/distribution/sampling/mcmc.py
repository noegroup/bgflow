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

    def forward(self, samples):
        proposal = [x + torch.randn_like(x) * self._noise_std for x in samples]
        delta_log_prob = 0.0  # symmetric density
        return proposal, delta_log_prob


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

    def forward(self, samples):
        *latent, logdet_inverse = self.flow.forward(*samples, inverse=True, **self.flow_kwargs)
        perturbed, delta_log_prob = self.base_proposal.forward(latent)
        *proposal, logdet_forward = self.flow.forward(*perturbed)
        # TODO: check if this needs to be the other way round   vvvv
        delta_log_prob = delta_log_prob - (logdet_forward + logdet_inverse)
        return list(proposal), delta_log_prob[:, 0]


class MCMCStep(SamplerStep):
    """Metropolis Monte Carlo"""
    def __init__(self, target_energy, proposal=GaussianProposal(), target_temperatures=1.0, n_steps=1):
        super().__init__(n_steps=n_steps)
        self.target_energy = target_energy
        self.target_temperatures = target_temperatures
        self.proposal = proposal

    def _step(self, state):
        # compute current energies
        if state.energies is None:
            state.energies = self.target_energy.energy(*state.samples)[..., 0]
        # make a proposal
        proposed_samples, delta_log_prob = self.proposal.forward(state.samples)
        proposed_energies = self.target_energy.energy(*proposed_samples)[..., 0]
        # accept according to Metropolis criterion
        accept = metropolis_accept(
            current_energies=state.energies/self.target_temperatures,
            proposed_energies=proposed_energies/self.target_temperatures,
            proposal_delta_log_prob=delta_log_prob
        )
        state.energies = torch.where(accept, proposed_energies, state.energies)
        state.samples = [torch.where(accept[..., None], new, old) for new, old in zip(proposed_samples, state.samples)]
        return state


class ApplyPeriodicBoundaries(SamplerStep):
    def __init__(self):
        super().__init__()

    def _step(self, state):
        if state.box_vectors is None:
            raise ValueError("Sampler state has no box vectors.")
        box_min, box_max = state.box_vectors
        for i in range(len(state.samples)):
            state.samples[i] = box_min + torch.fmod(state.samples[i], box_max - box_min)
        return state


def _GaussianMCMCSampler(
        energy,
        init_state=None,
        temperature=1.,
        noise_std=.1,
        n_stride=1,
        n_burnin=0,
        box_constraint=None,
        box_min=None, box_max=None
):
    state = SamplerState(samples=init_state, box_vector_min=box_min, box_vector_max=box_max)
    sampler_steps = [
        MCMCStep(
            energy,
            proposal=GaussianProposal(noise_std=noise_std),
            target_temperatures=temperature,
        ),
    ]
    if box_constraint is not None:
        raise ValueError("Use box_min and box_max instead.")
    if box_min is not None and box_max is not None:
        sampler_steps.append(ApplyPeriodicBoundaries())
    return IterativeSampler(
        initial_state=state,
        sampler_steps=sampler_steps,
        stride=n_stride,
        n_burnin=n_burnin
    )


class GaussianMCMCSampler(Energy, Sampler):
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
