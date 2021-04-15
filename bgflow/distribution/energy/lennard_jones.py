from .base import Energy
from bgflow.utils import distance_vectors, distances_from_vectors
import torch


__all__ = ["LennardJonesPotential"]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(Energy):
    def __init__(
            self, dim, n_particles, eps=1.0, rm=1.0, oscillator=True, oscillator_scale=1., two_event_dims=True):
        """Energy for a Lennard-Jones cluster

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim//n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = self.dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

    def _energy(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )
        dists = dists.view(-1, 1)

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        lj_energies = lj_energies.view(n_batch, -1).sum(-1) / 2

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(-1).sum(-1)
            return lj_energies + osc_energies * self._oscillator_scale
        else:
            return lj_energies

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x.view(-1, self._n_particles, self._n_dims)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()
