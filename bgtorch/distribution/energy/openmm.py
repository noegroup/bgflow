import warnings
import numpy as np
import torch

from multiprocessing import Pool, cpu_count

from ...utils.types import assert_numpy
from .base import Energy


_OPENMM_FLOATING_TYPE = np.float64
_SPATIAL_DIM = 3


class _OpenMMEnergyWrapper(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, openmm_energy_bridge):
        energy, force = openmm_energy_bridge.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None


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
    # TODO: cache context instead of recreating it
    from simtk import openmm
    global _openmm_context
    _openmm_context = openmm.Context(system, integrator, platform)


def _compute_energy_and_force(positions, err_handling):
    energy = None
    force = None

    try:
        # set positions
        _openmm_context.setPositions(positions)
        # compute state
        state = _openmm_context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy()
        force = state.getForces(asNumpy=True)

    except Exception as e:
        if err_handling == "warning":
            warnings.warn("Suppressed exception: {}".format(e))
        elif err_handling == "exception":
            raise e

    return energy, force


def _compute_energy_and_force_batch(args):
    """
    Parameters:
    -----------
    args : tuple
        A pair (positions, err_handling), where positions is an iterable of np.array and err_handling
        is one of {"ignore", "warning", "exception"}
    """
    positions, err_handling = args
    energies_and_forces = [_compute_energy_and_force(p, err_handling) for p in positions]
    return energies_and_forces


def _copy_openmm_integrator(integrator):
    """Copy an OpenMM integrator by serializing and deserializing it.
    The returned object is not bound to a Context.
    """
    from simtk import openmm
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(integrator))


class OpenMMEnergyBridge(object):
    """Bridge object to evaluate energies in OpenMM.
    Input positions are in nm, returned energies are dimensionless (units of kT), returned forces are in kT/nm.

    Parameters
    ----------
    openmm_system : simtk.openmm.System
        The OpenMM system object that contains all force objects.
    openmm_integrator : simtk.openmm.Integrator
        A thermostated OpenMM integrator (has to have a method `getTemperature()`.
    platform_name : str, optional
        An OpenMM platform name ('CPU', 'CUDA', 'Reference', or 'OpenCL')
    err_handling : str, optional
        How to handle infinite energies (one of {"warning", "ignore", "exception"}).
    n_workers : int, optional
        The number of processes used to compute energies in batches. This should not exceed the
         most-used batch size or the number of accessible CPU cores.
    """
    def __init__(self, openmm_system,
                 openmm_integrator,
                 platform_name='CPU', err_handling="warning", n_workers=1):

        from simtk import unit, openmm
        self._openmm_system = openmm_system
        self._openmm_integrator = openmm_integrator

        self._platform = openmm.Platform.getPlatformByName(platform_name)

        if platform_name == 'CPU':
            # Use only one thread/worker on the CPU platform
            # TODO: use [multiprocessing.cpu_count() // n_workers] instead of 1
            # TODO: maybe set n_workers to cpu_count by default
            self._platform.setPropertyDefaultValue('Threads', '1')
            # assert n_workers <= cpu_count()
            # self._platform.setPropertyDefaultValue('Threads', str(cpu_count()//n_workers))

        self.n_workers = n_workers
        if n_workers == 1:
            initialize_worker(openmm_system, self._openmm_integrator, self._platform)

        assert err_handling in ["ignore", "warning", "exception"]
        self._err_handling = err_handling

        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._unit_reciprocal = 1. / (openmm_integrator.getTemperature() * kB_NA)

    def _reduce_units(self, x):
        return x * self._unit_reciprocal

    def _simulate(self, n_steps):
        self._openmm_integrator.step(n_steps)

    def evaluate(self, batch):
        """batch: (B, N*D) """
        from simtk import unit

        # make a list of positions
        batch_array = assert_numpy(batch, arr_type=_OPENMM_FLOATING_TYPE)

        # assert correct number of positions
        assert batch_array.shape[1] == self._openmm_system.getNumParticles() * _SPATIAL_DIM

        # reshape to (B, N, D) and add physical units
        batch_array = unit.Quantity(value=batch_array.reshape(batch.shape[0], -1, _SPATIAL_DIM), unit=unit.nanometer)

        if self.n_workers == 1:
            energies_and_forces = _compute_energy_and_force_batch([batch_array, self._err_handling])
        else:
            # multiprocessing Pool
            pool = Pool(self.n_workers, initialize_worker,
                        (self._openmm_system, self._openmm_integrator, self._platform))

            # split list into equal parts
            chunks = np.array_split(batch_array, self.n_workers)
            args = [ (chunk, self._err_handling) for chunk in chunks ]

            energies_and_forces_ = pool.map(_compute_energy_and_force_batch, args)

            # concat lists
            energies_and_forces = []
            for ef in energies_and_forces_:
                energies_and_forces += ef
            pool.close()

        # remove units
        energies = [self._reduce_units(ef[0]) for ef in energies_and_forces]

        if not np.all(np.isfinite(energies)):
            if self._err_handling == "warning":
                warnings.warn("Infinite energy.")
            if self._err_handling == "exception":
                raise ValueError("Infinite energy.")

        # remove units
        forces = [np.ravel(self._reduce_units(ef[1]) * unit.nanometer) for ef in energies_and_forces]

        # to PyTorch tensors
        energies = torch.tensor(energies).to(batch).reshape(-1, 1)
        forces = torch.tensor(forces).to(batch)

        # store
        self.last_energies = energies
        self.last_forces = forces

        return energies, forces


class OpenMMEnergy(Energy):

    def __init__(self, dimension, openmm_energy_bridge):
        super().__init__(dimension)
        self._openmm_energy_bridge = openmm_energy_bridge
        self._last_batch = None

    def _energy(self, batch):
        # check if we have already computed this energy (hash of string representation should be sufficient)
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_energies
        else:
            self._last_batch = hash(str(batch))
            return _evaluate_openmm_energy(batch, self._openmm_energy_bridge)

    def force(self, batch, temperature=None):
        # check if we have already computed this energy
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_forces
        else:
            self._last_batch = hash(str(batch))
            return self._openmm_energy_bridge.evaluate(batch)[1]
