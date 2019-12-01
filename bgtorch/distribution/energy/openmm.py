import warnings
import numpy as np
import torch

from ...utils.types import assert_numpy
from .base import Energy


_OPENMM_FLOATING_TYPE = np.float64
_SPATIAL_DIM = 3


class _OpenMMEnergyWrapper(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, openmm_energy_bridge):
        energy, force = openmm_energy_bridge.evaluate(input, evaluate_forces=True)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None


_evaluate_openmm_energy = _OpenMMEnergyWrapper.apply


class OpenMMEnergyBridge(object):
    def __init__(
        self, 
        openmm_system, 
        openmm_integrator, 
        length_scale, 
        n_atoms=None, 
        openmm_integrator_args=None, 
        n_simulation_steps=0,
        err_handling="warning" 
    ):
        from simtk import openmm
        self._length_scale = length_scale
        self._openmm_integrator = openmm_integrator(*openmm_integrator_args)
        self._openmm_context = openmm.Context(openmm_system, self._openmm_integrator)
        self._n_simulation_steps = n_simulation_steps
        
        assert err_handling in ["ignore", "warning", "exception"]
        self._err_handling = err_handling 
        
        from simtk import unit
        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._unit_reciprocal = 1. / (self._openmm_integrator.getTemperature() * kB_NA)
    
    def _reduce_units(self, x):
        return x * self._unit_reciprocal
    
    def _assign_openmm_positions(self, configuration):
        from simtk.unit import Quantity
        positions = Quantity(
            value=configuration.reshape(-1, _SPATIAL_DIM), 
            unit=self._length_scale
        )
        self._openmm_context.setPositions(positions)
    
    def _get_energy_from_openmm_state(self, state):
        energy_quantity = state.getPotentialEnergy()
        return self._reduce_units(energy_quantity)
    
    def _get_force_from_openmm_state(self, state):
        forces_quantity = state.getForces(asNumpy=True)
        return np.ravel(self._reduce_units(forces_quantity) * self._length_scale)
    
    def _simulate(self, n_steps):
        self._openmm_integrator.step(n_steps)

    def _get_state(self, **kwargs):
        return self._openmm_context.getState(**kwargs)
    
    def _compute_energy_and_force(self, state, evaluate_energy=True, evaluate_force=True):
        self._assign_openmm_positions(state)

        if self._n_simulation_steps > 0:
            self._simulate(self._n_simulation_steps)

        state = self._get_state(getForces=evaluate_energy, getEnergy=evaluate_force)

        energy = None
        if evaluate_energy:
            energy = self._get_energy_from_openmm_state(state)

        force = None
        if evaluate_force:
            if np.isfinite(energy):
                    force = self._get_force_from_openmm_state(state)
            else:
                if self._err_handling == "warning":
                    warnings.warn("Computing gradients for infinite energies.")
                if self._err_handling == "exception":
                    raise ValueError("Computing gradients for infinite energies.")
        
        return energy, force

    def evaluate(self, batch, evaluate_energies=True, evaluate_forces=True):
        """batch: (B, N*D) """
        
        assert evaluate_energies or evaluate_forces,\
            "Cannot set both `evaluate_energies` and `evaluate_forces` to `False`."
        
        n_batch = batch.shape[0]
        
        states = assert_numpy(batch, arr_type=_OPENMM_FLOATING_TYPE)
        
        forces = None
        if evaluate_forces:
            forces = np.zeros_like(states)

        energies = None
        if evaluate_energies:
            energies = np.zeros((n_batch, 1), dtype=states.dtype)
        
        for batch_idx, state in enumerate(states):
            try:
                energy, force = self._compute_energy_and_force(
                    state, evaluate_energies, evaluate_forces)
                if energy is not None:
                    energies[batch_idx] = energy
                if forces is not None:
                    forces[batch_idx] = force 
            except Exception as e:
                if self._err_handling == "warning":
                    warnings.warn("Suppressed exception: {}".format(e))
                elif self._err_handling == "exception":
                    raise e
        
        return torch.Tensor(energies).to(batch), torch.Tensor(forces).to(batch)


class OpenMMEnergy(Energy):

    def __init__(self, dimension, openmm_energy_bridge):
        super().__init__(dimension)
        self._openmm_energy_bridge = openmm_energy_bridge

    def _energy(self, state, no_grads=False):
        if no_grads:
            return torch.Tensor(
                self._openmm_energy_bridge.evaluate(state, evaluate_forces=False)[0]
            ).to(state)
        else:
            return _evaluate_openmm_energy(state, self._openmm_energy_bridge)

    def _force(self, state):
        return torch.tensor(
            self._openmm_energy_bridge.evaluate(state, evaluate_forces=True)[1]
        ).to(state)