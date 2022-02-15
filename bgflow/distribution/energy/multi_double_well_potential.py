from .base import Energy
from bgflow.utils import compute_distances

__all__ = ["MultiDoubleWellPotential"]


class MultiDoubleWellPotential(Energy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)
