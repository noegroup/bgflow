import torch
import numpy as np

from .base import EnergyBasedModel


# TODO: write docstrings


class BoxedEnergy(EnergyBasedModel):
    
    def __init__(self, orig_system, temperature=1., min_val=-1., max_val=1.):
        self._orig_system = orig_system
        self._min_val = min_val
        self._max_val = max_val
    
    def sample(self, sample_shape, temperature=None):
        x = self._orig_system.sample(sample_shape, temperature)
        x = torch.sigmoid(x) * (self._max_val - self._min_val) + self._min_val
        return x
    
    def energy(self, x, temperature=None):
        energy = self._orig_system.energy(x, temperature)
        dlogp = (torch.nn.functional.logsigmoid(x) - torch.nn.functional.softplus(x)).sum(dim=-1, keepdim=True)
        energy = energy + dlogp
        return energy
