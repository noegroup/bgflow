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
        # energy, force = openmm_energy_bridge.evaluate(input, evaluate_forces=True)
        energy, force = openmm_energy_bridge.evaluate(input)
        if openmm_energy_bridge.evaluate_force:
            ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        try:
            neg_force, = ctx.saved_tensors
            grad_input = grad_output * neg_force
            return grad_input, None
        except Exception as e:
            print('Could not compute backward pass. Make sure that evaluate_force=True is set in OpenMMEnergy')
            raise e


_evaluate_openmm_energy = _OpenMMEnergyWrapper.apply


# Use multiprocessing.Pool to create workers that manage their own Context objects
def initialize_worker(system, integrator, platform):
    """Initialize a multiprocessing.Pool worker by caching a Context object

    Parameters
    ----------
    system : simtk.openmm.System
        The System object for which energies are to be computed
    platform_name : str
        The platform name to open the Context for
    """
    from simtk import openmm
    global _openmm_context
    _openmm_context = openmm.Context(system, integrator, platform)


def _compute_energy_and_force(args):
    positions, getEnergy, getForces, _err_handling = args
    energy = None
    force = None

    try:
        # set positions
        _openmm_context.setPositions(positions)
        # compute state
        state = _openmm_context.getState(getEnergy=getEnergy, getForces=getForces)

        if getEnergy:
            energy = state.getPotentialEnergy()
        if getForces:
            force = state.getForces(asNumpy=True)

    except Exception as e:
        if _err_handling == "warning":
            warnings.warn("Suppressed exception: {}".format(e))
        elif _err_handling == "exception":
            raise e

    return energy, force


class OpenMMEnergyBridge(object):
    def __init__(self, openmm_system, length_scale,
                 openmm_integrator=None, openmm_integrator_args=None, n_simulation_steps=0,
                 platform_name='CPU', err_handling="warning", n_workers=1,
                 evaluate_energy=True, evaluate_force=True):
        from simtk import openmm
        self._openmm_system = openmm_system
        self._length_scale = length_scale

        assert evaluate_energy or evaluate_force, "Either `evaluate_energy` or `evaluate_force` must be `True`."
        self.evaluate_energy = evaluate_energy
        self.evaluate_force = evaluate_force

        if openmm_integrator is None:
            self._openmm_integrator = openmm.VerletIntegrator(0.001)
        else:
            self._openmm_integrator = openmm_integrator(*openmm_integrator_args)

        self._platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == 'CPU':
            # Use only one thread/worker on the CPU platform
            self._platform.setPropertyDefaultValue('Threads', '1')

        self.n_workers = n_workers
        if n_workers == 1:
            initialize_worker(openmm_system, self._openmm_integrator, self._platform)

        # self._openmm_context = openmm.Context(openmm_system, self._openmm_integrator, platform)
        self._n_simulation_steps = n_simulation_steps
        
        assert err_handling in ["ignore", "warning", "exception"]
        self._err_handling = err_handling 
        
        from simtk import unit
        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._unit_reciprocal = 1. / (self._openmm_integrator.getTemperature() * kB_NA)

    def initialize_worker(self):
        """Initialize a multiprocessing.Pool worker by caching a Context object

        Parameters
        ----------
        system : simtk.openmm.System
            The System object for which energies are to be computed
        platform_name : str
            The platform name to open the Context for
        """
        from simtk import openmm
        # global context
        self._openmm_context = openmm.Context(self._system, self._openmm_integrator, self._platform)
        # return context

    def _reduce_units(self, x):
        return x * self._unit_reciprocal

    def _simulate(self, n_steps):
        self._openmm_integrator.step(n_steps)

    def evaluate(self, batch):
        """batch: (B, N*D) """

        n_batch = batch.shape[0]

        # make a list of positions
        batch_array = assert_numpy(batch, arr_type=_OPENMM_FLOATING_TYPE)
        batch_positions = []
        from simtk.unit import Quantity
        for positions in batch_array:
            # set unit
            batch_positions.append(Quantity(value=positions.reshape(-1, _SPATIAL_DIM), unit=self._length_scale))

        if self.n_workers == 1:
            energies_and_forces = [_compute_energy_and_force((positions,
                                                              self.evaluate_energy,
                                                              self.evaluate_force,
                                                              self._err_handling)) for positions in batch_positions]
        else:
            from multiprocessing import Pool
            # Create a multiprocessing pool of workers that cache a Context object for the System being evaluated
            self.pool = Pool(self.n_workers, initialize_worker,
                             (self._openmm_system, self._openmm_integrator, self._platform))

            args = [(pos, self.evaluate_energy, self.evaluate_force, self._err_handling) for pos in batch_positions]
            # Compute energies and forces
            energies_and_forces = self.pool.map(_compute_energy_and_force, args)
            # Shut down workers
            self.pool.close()

        if self.evaluate_energy:
            energies = [self._reduce_units(ef[0]) for ef in energies_and_forces]
        else:
            energies = np.zeros((n_batch, 1), dtype=batch_array.dtype)

        if not np.all(np.isfinite(energies)):
            if self._err_handling == "warning":
                warnings.warn("Infinite energy.")
            if self._err_handling == "exception":
                raise ValueError("Infinite energy.")

        if self.evaluate_force:
            forces = [np.ravel(self._reduce_units(ef[1]) * self._length_scale) for ef in energies_and_forces]
        else:
            forces = np.zeros_like(batch)

        return torch.tensor(energies).to(batch).reshape(-1, 1), torch.tensor(forces).to(batch)


class OpenMMEnergy(Energy):

    def __init__(self, dimension, openmm_energy_bridge):
        super().__init__(dimension)
        self._openmm_energy_bridge = openmm_energy_bridge

    def _activate_energies(self):
        if not self._openmm_energy_bridge.evaluate_energy:
            self._openmm_energy_bridge.evaluate_energy = True
            warnings.warn("OpenMM initialized without energy evaluation, but energy evaluation was just switched on. " +
                          "This will slow down calculation")

    def _activate_forces(self):
        if not self._openmm_energy_bridge.evaluate_force:
            self._openmm_energy_bridge.evaluate_force = True
            warnings.warn("OpenMM initialized without force evaluation, but force evaluation was just switched on. " +
                          "This will slow down calculation")

    def _energy(self, batch):
        self._activate_energies()
        return _evaluate_openmm_energy(batch, self._openmm_energy_bridge)
        # if no_grads:
        #     return self._openmm_energy_bridge.evaluate(batch, evaluate_forces=False)[0]
        # else:
        #     self._activate_forces()
        #     return _evaluate_openmm_energy(batch, self._openmm_energy_bridge)

    def force(self, batch, temperature=None):
        self._activate_forces()
        return self._openmm_energy_bridge.evaluate(batch)[1]
