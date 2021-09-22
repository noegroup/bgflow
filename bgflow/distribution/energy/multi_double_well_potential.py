from .base import Energy
from bgflow.utils import distance_vectors, distances_from_vectors
import torch


__all__ = ["MultiDoubleWellPotential"]


class MultiDoubleWellPotential(Energy):
    
    def __init__(self, dim, n_particles, a, b, c, offset):
        super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset
        
    @property
    def dim(self):
        return self._dim
    
    def energy_dw(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, -1)

        dists = distances_from_vectors(
            distance_vectors(x.view(n_batch, self._n_particles, -1))
        )
        dists = dists.view(-1, 1)

        dists = dists - self._offset

        energies = self._a * dists**4 + self._b * dists**2 + self._c
        return energies.view(n_batch, -1).sum(-1) / 2
        
    def _energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self.energy_dw(x).view(-1, 1) / temperature
    
    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self._energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]
