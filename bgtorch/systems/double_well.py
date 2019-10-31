import torch
import numpy as np

from .base import EnergyBasedModel
from .harmonic_oscillator import _harmonic_potential


# TODO: write docstrings


def _double_well_potential(x, a=0.0, b=-4.0, c=1.0):
    return a * x + b * x.pow(2) + c * x.pow(4)


class DoubleWell(EnergyBasedModel):
    def __init__(self, n_dw_dims=1, a=0.0, b=-4.0, c=1.0, sigma=1.0, temperature=1.0):
        self._n_dw_dims = n_dw_dims
        self._a = a
        self._b = b
        self._c = c
        self._sigma = sigma
        self._temperature = temperature

    def energy(self, x, temperature=None):
        if temperature is None:
            temperature = self._temperature

        dw_component = x[..., : self._n_dw_dims]
        dw_energy = _double_well_potential(dw_component, self._a, self._b, self._c)

        harmonic_component = x[..., self._n_dw_dims :]
        harmonic_energy = _harmonic_potential(harmonic_component, self._sigma)

        return (dw_energy + harmonic_energy).sum(dim=-1, keepdim=True) / temperature
