import torch
import numpy as np

from .energy import Energy
from .sampling import Sampler
from ..utils.types import assert_numpy


__all__ = ["MixtureDistribution"]


class MixtureDistribution(Energy, Sampler):
    def __init__(self, components, unnormed_log_weights=None, trainable_weights=False):
        assert all([c.dim == components[0].dim for c in components]),\
            "All mixture components must have the same dimensionality."
        super().__init__(components[0].dim)
        self._components = torch.nn.ModuleList(components)
        
        if unnormed_log_weights is None:
            unnormed_log_weights = torch.zeros(len(components))
        else:
            assert len(unnormed_log_weights.shape) == 1,\
                "Mixture weights must be a Tensor of shape `[n_components]`."
            assert len(unnormed_log_weights) == len(components),\
                "Number of mixture weights does not match number of components."
        if trainable_weights:
            self._unnormed_log_weights = torch.nn.Parameter(unnormed_log_weights)
        else:
            self.register_buffer("_unnormed_log_weights", unnormed_log_weights)
    
    @property
    def _log_weights(self):
        return torch.log_softmax(self._unnormed_log_weights, dim=-1)
    
    def _sample(self, n_samples):
        weights_numpy = assert_numpy(self._log_weights.exp())
        ns = np.random.multinomial(n_samples, weights_numpy, 1)[0]
        samples = [c.sample(n) for n, c in zip(ns, self._components)]
        return torch.cat(samples, dim=0)
        
    def _energy(self, x):
        energies = torch.stack([c.energy(x) for c in self._components], dim=-1)
        return -torch.logsumexp(-energies + self._log_weights.view(1, 1, -1), dim=-1)
    
    def _log_assignments(self, x):
        energies = torch.stack([c.energy(x) for c in self._components], dim=-1)
        return -energies