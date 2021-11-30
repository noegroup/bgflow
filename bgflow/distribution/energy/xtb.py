"""Wrapper for semi-empirical QM energies with XTB.
"""

__all__ = ["XTBEnergy", "XTBBridge"]


import warnings
import torch
import numpy as np
from .base import Energy
from ...utils.types import assert_numpy


_XTB_FLOATING_TYPE = np.float64
_SPATIAL_DIM = 3


class _XTBEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, xtb_energy_bridge):
        energy, force, *_ = xtb_energy_bridge.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None

_evaluate_xtb_energy = _XTBEnergyWrapper.apply


class XTBBridge:
    """Wrapper around XTB for semi-empirical QM energy calculations.

    Parameters
    ----------
    numbers : np.ndarray
        Atomic numbers
    temperature : float
        Temperature in Kelvin.
    method : str
        The semi-empirical method that is used to compute energies.
    solvent : str
        The solvent. If empty string, perform a vacuum calculation.
    verbosity : int
        0 (muted), 1 (minimal), 2 (full)
    err_handling : str
        How to deal with exceptions inside XTB. One of `["ignore", "warning", "error"]`

    Attributes
    ----------
    n_atoms : int
        The number of atoms in this molecules.
    available_solvents : List[str]
        The solvent models that are available for computations in xtb.
    available_methods : List[str]
        The semiempirical methods that are available for computations in xtb.

    Examples
    --------
    Setting up an XTB energy for a small peptide from bgmol
    >>> from bgmol.systems import MiniPeptide
    >>> from bgflow import XTBEnergy, XTBBridge
    >>> import numpy as np
    >>> import torch
    >>> system = MiniPeptide("G")
    >>> numbers = np.array([atom.element.number for atom in system.mdtraj_topology.atoms])
    >>> target = XTBEnergy(XTBBridge(numbers=numbers, temperature=300, solvent="water"))
    >>> xyz = torch.tensor(system.positions)
    >>> energy = target.energy(xyz)

    Notes
    -----
    Requires the xtb-python program (installable with `conda install -c conda-forge xtb-python`).

    """
    def __init__(
            self,
            numbers: np.ndarray,
            temperature: float,
            method: str = "GFN2-xTB",
            solvent: str = "",
            verbosity: int = 0,
            err_handling: str = "warning"
    ):
        self.numbers = numbers
        self.temperature = temperature
        self.method = method
        self.solvent = solvent
        self.verbosity = verbosity
        self.err_handling = err_handling
        self._last_batch = None
        self.last_energies = None
        self.last_forces = None

    @property
    def n_atoms(self):
        return len(self.numbers)

    @property
    def available_solvents(self):
        from xtb.utils import _solvents
        return list(_solvents.keys())

    @property
    def available_methods(self):
        from xtb.utils import _methods
        return list(_methods.keys())

    def evaluate(self, positions: torch.Tensor):
        self._last_batch = hash(str(positions))

        shape = positions.shape
        assert shape[-2:] == (self.n_atoms, 3) or shape[-1] == self.n_atoms * 3
        energy_shape = shape[:-2] if shape[-2:] == (self.n_atoms, 3) else shape[:-1]
        # the stupid last dim
        energy_shape = [*energy_shape, 1]
        position_batch = assert_numpy(positions.reshape(-1, self.n_atoms, 3), arr_type=_XTB_FLOATING_TYPE)

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
        from xtb.interface import Calculator, XTBException
        from xtb.utils import get_method, get_solvent
        positions = _nm2bohr(positions)
        try:
            calc = Calculator(get_method(self.method), self.numbers, positions)
            calc.set_solvent(get_solvent(self.solvent))
            calc.set_verbosity(self.verbosity)
            calc.set_electronic_temperature(self.temperature)
            try:
                res = calc.singlepoint()
            except XTBException:
                # Try with higher temperature
                calc.set_electronic_temperature(10 * self.temperature)
                res = calc.singlepoint()
                calc.set_electronic_temperature(self.temperature)
                res = calc.singlepoint(res)
            energy = res.get_energy()
            force = -res.get_gradient()
            assert not np.isnan(energy)
            assert not np.isnan(force).any()
        except XTBException as e:
            if self.err_handling == "error":
                raise e
            elif self.err_handling == "warning":
                warnings.warn(
                    f"Caught exception in xtb. "
                    f"Returning infinite energy and zero force. "
                    f"Original exception: {e}"
                )
                force = np.zeros_like(positions)
                energy = np.infty
            elif self.err_handling == "ignore":
                force = np.zeros_like(positions)
                energy = np.infty
        except AssertionError:
            force[np.isnan(force)] = 0.
            energy = np.infty
            if self.err_handling in ["error", "warning"]:
                warnings.warn("Found nan in xtb force or energy. Returning infinite energy and zero force.")

        energy = _hartree2kbt(energy, self.temperature)
        force = _hartree_per_bohr2kbt_per_nm(force, self.temperature)
        return energy, force


class XTBEnergy(Energy):
    """Semi-empirical energy computation with XTB.

    Parameters
    ----------
    xtb_bridge : XTBBridge
        The wrapper object.
    two_event_dims : bool
        Whether to use two event dimensions.
        In this case, the energy call expects positions of shape (*batch_shape, n_atoms, 3).
        Otherwise, it expects positions of shape (*batch_shape, n_atoms * 3).
    """
    def __init__(self, xtb_bridge: XTBBridge, two_event_dims=True):
        event_shape = (xtb_bridge.n_atoms, 3) if two_event_dims else (xtb_bridge.n_atoms * 3, )
        super().__init__(event_shape)
        self._xtb_bridge = xtb_bridge
        self._last_batch = None

    def _energy(self, batch, no_grads=False):
        # check if we have already computed this energy (hash of string representation should be sufficient)
        if hash(str(batch)) == self._last_batch:
            return self._xtb_bridge.last_energies
        else:
            self._last_batch = hash(str(batch))
            return _evaluate_xtb_energy(batch, self._xtb_bridge)

    def force(self, batch, temperature=None):
        # check if we have already computed this energy
        if hash(str(batch)) == self._last_batch:
            return self._xtb_bridge.last_forces
        else:
            self._last_batch = hash(str(batch))
            return self._xtb_bridge.evaluate(batch)[1]


_BOLTZMANN_CONSTANT_HE = 3.1668115634556076e-06  # in hartree / kelvin
_BOHR_RADIUS = 0.0529177210903  # nm


def _bohr2nm(x):
    return x * _BOHR_RADIUS


def _nm2bohr(x):
    return x / _BOHR_RADIUS


def _per_bohr2per_nm(x):
    return _nm2bohr(x)


def _hartree2kbt(x, temperature):
    kbt = _BOLTZMANN_CONSTANT_HE * temperature
    return x / kbt


def _hartree_per_bohr2kbt_per_nm(x, temperature):
    return _per_bohr2per_nm(_hartree2kbt(x, temperature))
