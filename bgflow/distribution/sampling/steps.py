"""Iterative Sampler steps
"""

import warnings
import torch
from .iterative import SamplerStep


class ReplicaExchangeStep(SamplerStep):
    def __init__(self, energy, temperature_scalings):
        super().__init__()
        # TODO
        self.energy = energy
        self.temperature_scalings = temperature_scalings
        self.is_odd_step = False

    def _step(self, state):
        if state.energies is None:
            # TODO: check if energies are up to date
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
        # perform swaps
        # TODO
        # rescale velocities
        if state.momenta is not None:
            # TODO
            pass
        return state

