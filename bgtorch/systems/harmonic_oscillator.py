import torch
import numpy as np

from .base import EnergyBasedModel

# TODO: write docstrings


def _harmonic_potential(x, sigma=1.0):
    return 0.5 * x.pow(2) / sigma


class HarmonicOscillator(EnergyBasedModel):
    def __init__(self, n_dims, temperature=1.0):
        self._n_dims = n_dims
        self._temperature = temperature

    def sample(self, sample_shape, temperature=None):
        if temperature is None:
            temperature = self._temperature
        if isinstance(temperature, torch.Tensor):
            scale = torch.sqrt(temperature)
        else:
            scale = np.sqrt(temperature)
        return torch.Tensor(*sample_shape, self._n_dims).normal_() * scale

    def energy(self, x, temperature=None):
        if temperature is None:
            temperature = self._temperature
        return _harmonic_potential(x).sum(dim=-1, keepdim=True) / temperature
