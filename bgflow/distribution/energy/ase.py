"""Wrapper around ASE (atomic simulation environment)
"""
__all__ = ["ASEBridge", "ASEEnergy"]


import warnings
import torch
import numpy as np
from ...utils import assert_numpy
from .base import _BridgeEnergy, _Bridge


class ASEBridge(_Bridge):
    """Wrapper around Atomic Simulation Environment.

    Parameters
    ----------
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
        assert hasattr(atoms, "calc")
        self.atoms = atoms
        self.temperature = temperature
        self._last_batch = None
        self.err_handling = err_handling
        self.last_energies = None
        self.last_forces = None

    @property
    def n_atoms(self):
        return len(self.atoms)

    def evaluate(self, positions: torch.Tensor):
        shape = positions.shape
        assert shape[-2:] == (self.n_atoms, 3) or shape[-1] == self.n_atoms * 3
        energy_shape = shape[:-2] if shape[-2:] == (self.n_atoms, 3) else shape[:-1]
        # the stupid last dim
        energy_shape = [*energy_shape, 1]
        position_batch = assert_numpy(positions.reshape(-1, self.n_atoms, 3), arr_type=self._FLOATING_TYPE)

        energy_batch = np.zeros(energy_shape, dtype=position_batch.dtype)
        force_batch = np.zeros_like(position_batch)

        for i, pos in enumerate(position_batch):
            energy_batch[i], force_batch[i] = self._evaluate_single(pos)

        energies = torch.tensor(energy_batch.reshape(*energy_shape)).to(positions)
        forces = torch.tensor(force_batch.reshape(*shape)).to(positions)

        # store
        self.last_energies = energies
        self.last_forces = forces

        return energies, forces

    def _evaluate_single(self, positions):
        from ase.units import kB, nm
        try:
            self.atoms.positions = positions * nm
            energy = self.atoms.get_potential_energy()
            force = self.atoms.get_forces()
            assert not np.isnan(energy)
            assert not np.isnan(force).any()
        except AssertionError:
            force[np.isnan(force)] = 0.
            energy = np.infty
            if self.err_handling in ["error", "warning"]:
                warnings.warn("Found nan in ase force or energy. Returning infinite energy and zero force.")
        kbt = kB * 300
        energy = energy / kbt
        force = force / (kbt / nm)
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
