import torch
import numpy as np

from .base import EnergyBasedModel


# TODO write docstrings


class UniformEnergy(EnergyBasedModel):
    def __init__(self, n_dims, temperature=1.0, min_val=-1.0, max_val=1.0):
        self._n_dims = n_dims
        self._temperature = temperature
        self._min_val = min_val
        self._max_val = max_val

    def sample(self, sample_shape, temperature=None):
        return torch.Tensor(*sample_shape, self._n_dims).uniform_(
            self._min_val, self._max_val
        )

    def energy(self, x, temperature=None):
        # uniform has constant energy
        return torch.zeros(x.shape[0], 1)
