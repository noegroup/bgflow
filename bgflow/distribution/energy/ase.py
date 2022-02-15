"""Wrapper around ASE (atomic simulation environment)
"""
__all__ = ["ASEBridge", "ASEEnergy"]


import warnings
import torch
import numpy as np
from .base import _BridgeEnergy, _Bridge


class ASEBridge(_Bridge):
    """Wrapper around Atomic Simulation Environment.

    Parameters
    ----------
    atoms : ase.Atoms
        An `Atoms` object that has a calculator attached to it.
    temperature : float
        Temperature in Kelvin.
    err_handling : str
        How to deal with exceptions inside ase. One of `["ignore", "warning", "error"]`

    Notes
    -----
    Requires the ase package (installable with `conda install -c conda-forge ase`).

    """
    def __init__(
            self,
            atoms,
            temperature: float,
            err_handling: str = "warning"
    ):
        super().__init__()
        assert hasattr(atoms, "calc")
        self.atoms = atoms
        self.temperature = temperature
        self.err_handling = err_handling

    @property
    def n_atoms(self):
        return len(self.atoms)

    def _evaluate_single(
            self,
            positions: torch.Tensor,
            evaluate_force=True,
            evaluate_energy=True,
    ):
        from ase.units import kB, nm
        kbt = kB * self.temperature
        energy, force = None, None
        try:
            self.atoms.positions = positions * nm
            if evaluate_energy:
                energy = self.atoms.get_potential_energy() / kbt
            if evaluate_force:
                force = self.atoms.get_forces() / (kbt / nm)
            assert not np.isnan(energy)
            assert not np.isnan(force).any()
        except AssertionError as e:
            force[np.isnan(force)] = 0.
            energy = np.infty
            if self.err_handling == "warning":
                warnings.warn("Found nan in ase force or energy. Returning infinite energy and zero force.")
            elif self.err_handling == "error":
                raise e
        return energy, force


class ASEEnergy(_BridgeEnergy):
    """Energy computation with calculators from the atomic simulation environment (ASE).
    Various molecular simulation programs provide wrappers for ASE,
    see https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
    for a list of available calculators.

    Examples
    --------
    Use the calculator from the xtb package to compute the energy of a water molecule with the GFN2-xTB method.
    >>> from ase.build import molecule
    >>> from xtb.ase.calculator import XTB
    >>> water = molecule("H2O")
    >>> water.calc = XTB()
    >>> target = ASEEnergy(ASEBridge(water, 300.))
    >>> pos = torch.tensor(0.1*water.positions, **ctx)
    >>> energy = target.energy(pos)

    Parameters
    ----------
    ase_bridge : ASEBridge
        The wrapper object.
    two_event_dims : bool
        Whether to use two event dimensions.
        In this case, the energy call expects positions of shape (*batch_shape, n_atoms, 3).
        Otherwise, it expects positions of shape (*batch_shape, n_atoms * 3).
    """
    pass
