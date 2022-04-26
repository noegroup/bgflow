from ...utils.train import linlogcut, ClipGradient
from .base import Energy


__all__ = ["LinLogCutEnergy", "GradientClippedEnergy"]


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


class GradientClippedEnergy(Energy):
    """An Energy with clipped gradients. See `ClipGradient` for details."""
    def __init__(self, energy: Energy, gradient_clipping: ClipGradient):
        super().__init__(energy.event_shapes)
        self.delegate = energy
        self.clipping = gradient_clipping

    def _energy(self, *xs, **kwargs):
        return self.delegate.energy(*((self.clipping(x) for x in xs)), **kwargs)
