from ...utils.train import linlogcut
from .base import Energy

__all__ = ["LinLogCutEnergy"]


class LinLogCutEnergy(Energy):
    """Cut off energy at singularities.

    Parameters
    ----------
    energy : Energy
    high_energy : float
        Energies beyond this value are replaced by `u = high_energy + log(1 + energy - high_energy)`
    max_energy : float
        Upper bound for energies returned by this object.
    """
    def __init__(self, energy, high_energy=1e3, max_energy=1e9):
        super().__init__(energy.event_shapes)
        self.delegate = energy
        self.high_energy = high_energy
        self.max_energy = max_energy

    def _energy(self, *xs, **kwargs):
        u = self.delegate.energy(*xs, **kwargs)
        return linlogcut(u, high_val=self.high_energy, max_val=self.max_energy)
