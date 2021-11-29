"""Wrapper for semi-empirical QM energies with XTB.
"""

__all__ = ["XTBEnergy", "XTBBridge"]

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
    method : str
        The semi-empirical method that is used to compute energies
    solvent : str
        The solvent.
    verbosity : int
        0 (muted), 1 (minimal), 2 (full)

    Notes
    -----
    Requires the xtb-python program (installable with `conda install -c conda-forge xtb-python`).

    """
    def __init__(
            self,
            numbers: np.ndarray = None,
            method: str = "GFN2-xTB",
            solvent: str = "water",
            verbosity: int = 0
    ):
        self.method = method
        if numbers is not None:
            self.numbers = numbers
        self.solvent = solvent
        self.verbosity = verbosity
        self._last_batch = None

    @property
    def n_atoms(self):
        return len(self.numbers)

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
            energy_batch[i, 0], force_batch[i] = self._evaluate_single(pos)

        energies = torch.tensor(energy_batch.reshape(*energy_shape)).to(positions)
        forces = torch.tensor(force_batch.reshape(*shape)).to(positions)

        # store
        self.last_energies = energies
        self.last_forces = forces

        return energies, forces

    def _evaluate_single(self, positions):
        from xtb.interface import Calculator
        from xtb.utils import get_method, get_solvent
        calc = Calculator(get_method(self.method), self.numbers, positions)
        calc.set_solvent(get_solvent(self.solvent))
        calc.set_verbosity(self.verbosity)
        res = calc.singlepoint()
        return res.get_energy(), -res.get_gradient()


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
