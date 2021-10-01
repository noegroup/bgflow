"""Energy and Force computation in OpenMM
"""

import warnings
import multiprocessing as mp
from collections import namedtuple

import numpy
import numpy as np
import pickle
import torch

from ...utils.types import assert_numpy
from .base import Energy

__all__ = ["OpenMMBridge", "OpenMMEnergy"]


_OPENMM_FLOATING_TYPE = np.float64
_SPATIAL_DIM = 3


class _OpenMMEnergyWrapper(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, openmm_energy_bridge):
        energy, force, *_ = openmm_energy_bridge.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None


_evaluate_openmm_energy = _OpenMMEnergyWrapper.apply


class OpenMMEnergy(Energy):
    """Energy of a molecular system computed with openmm.

    Parameters
    ----------
    dimension : int
        Dimension of the configurational space (num_particles * 3).
    openmm_energy_bridge : OpenMMBridge
        This instance delegates computations to OpenMM using multiprocessing.
    """
    def __init__(self, dimension, openmm_energy_bridge):
        super().__init__(dimension)
        self._openmm_energy_bridge = openmm_energy_bridge
        self._last_batch = None

    def _energy(self, batch, no_grads=False):
        # check if we have already computed this energy (hash of string representation should be sufficient)
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_output.energies
        else:
            self._last_batch = hash(str(batch))
            return _evaluate_openmm_energy(batch, self._openmm_energy_bridge)

    def force(self, batch, temperature=None):
        # check if we have already computed this energy
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_output.forces
        else:
            self._last_batch = hash(str(batch))
            return self._openmm_energy_bridge.evaluate(batch)[1]


class OpenMMBridge:
    """Bridge object to evaluate energies in OpenMM.
    Input positions are in nm, returned energies are dimensionless (units of kT), returned forces are in kT/nm.

    Parameters
    ----------
    openmm_system : openmm.System
        The OpenMM system object that contains all force objects.
    openmm_integrator : openmm.Integrator
        A thermostated OpenMM integrator (has to have a method `getTemperature()`.
    platform_name : str, optional
        An OpenMM platform name ('CPU', 'CUDA', 'Reference', or 'OpenCL')
    err_handling : str, optional
        How to handle infinite energies (one of {"warning", "ignore", "exception"}).
    n_workers : int, optional
        The number of processes used to compute energies in batches. This should not exceed the
        most-used batch size or the number of accessible CPU cores. The default is the number
        of logical cpu cores. If a GPU platform is used (CUDA or OpenCL), n_workers is always set to 1
        to sidestep multiprocessing (due to performance issues when using multiprocessing with GPUs).
    n_simulation_steps : int, optional
        If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.
    """
    def __init__(
        self,
        openmm_system,
        openmm_integrator,
        platform_name='CPU',
        err_handling="warning",
        n_workers=mp.cpu_count(),
        n_simulation_steps=0,
    ):
        try:
            from openmm import unit
        except ImportError:
            from simtk import unit   # old openmm versions

        platform_properties = {'Threads': str(max(1, mp.cpu_count()//n_workers))} if platform_name == "CPU" else {}

        # Compute all energies in child processes due to a bug in the OpenMM's PME code.
        # This might be problematic if an energy has already been computed in the same program on the parent thread,
        # see https://github.com/openmm/openmm/issues/2602
        self._openmm_system = openmm_system
        self._openmm_integrator = openmm_integrator
        if platform_name in ["CUDA", "OpenCL"] or n_workers == 1:
            self.context_wrapper = SingleContext(
                1, openmm_system, openmm_integrator, platform_name, platform_properties
            )
        else:
            self.context_wrapper = MultiContext(
                n_workers, openmm_system, openmm_integrator, platform_name, platform_properties
            )
        self._err_handling = err_handling
        self._n_simulation_steps = n_simulation_steps
        self.last_output = None

    @property
    def integrator(self):
        return self._openmm_integrator

    @property
    def n_simulation_steps(self):
        return self._n_simulation_steps

    def evaluate(
            self,
            batch,
            box_vector_batch=None,
            velocity_batch=None,
            temperature_scaling_batch=None,
            evaluate_force=True,
            evaluate_energy=True,
            evaluate_positions=False,
            evaluate_box_vectors=False,
            evaluate_velocities=False,
            evaluate_path_probability_ratio=False
    ):
        """
        Compute energies/forces for a batch of positions.

        Parameters:
        -----------
        batch : np.ndarray or torch.Tensor
            A batch of particle positions that has shape (batch_size, num_particles * 3).
        box_vector_batch : np.ndarray or torch.Tensor, optional
            A batch of box vectors that has shape (batch_size, 3, 3).
        velocity_batch : np.ndarray or torch.Tensor, optional
            A batch of particle positions that has shape (batch_size, num_particles * 3).
        temperature_scaling_batch :  np.ndarray or torch.Tensor, optional
            A batch of temperature scaling factors to be applied indididually to each state.
        evaluate_force : bool, optional
            Whether to compute forces.
        evaluate_energy : bool, optional
            Whether to compute energies.
        evaluate_positions : bool, optional
            Whether to return positions.
        evaluate_velocities : bool, optional
            Whether to return velocities.
        evaluate_path_probability_ratio : bool, optional
            Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.

        Returns
        -------
        result : _OpenMMOutput
            A namedtuple of results from the openmm computation, where forces are batched torch tensors
            of shape (batchsize, n_particles * 3).
        """
        # check shapes
        n_particles = self._openmm_system.getNumParticles()
        sample_shape = batch.shape[:-1] if batch.shape[-1] == n_particles * 3 else batch.shape[:-2]
        assert box_vector_batch is None or box_vector_batch.shape == (*sample_shape, 3, 3)
        assert velocity_batch is None or velocity_batch.shape == batch.shape
        assert temperature_scaling_batch is None or temperature_scaling_batch.shape == (*sample_shape, )

        # wrap into input struct
        def to_numpy_or_none(x, shape):
            if x is None:
                return None
            else:
                return assert_numpy(x.reshape(shape), arr_type=_OPENMM_FLOATING_TYPE)

        openmm_input = _OpenMMInput(
            positions=to_numpy_or_none(batch, (-1, n_particles, _SPATIAL_DIM)),
            box_vectors=to_numpy_or_none(box_vector_batch, (-1, 3, 3)),
            velocities=to_numpy_or_none(velocity_batch, (-1, n_particles, _SPATIAL_DIM)),
            temperature_scaling=to_numpy_or_none(temperature_scaling_batch, (-1,)),
            evaluate_energy=evaluate_energy,
            evaluate_force=evaluate_force,
            evaluate_positions=evaluate_positions,
            evaluate_velocities=evaluate_velocities,
            evaluate_box_vectors=evaluate_box_vectors,
            evaluate_path_probability_ratio=evaluate_path_probability_ratio,
            err_handling=self._err_handling,
            n_simulation_steps=self._n_simulation_steps
        )

        # call openmm
        result = self.context_wrapper.evaluate(openmm_input)

        # convert to PyTorch tensors

        def to_tensor(x, condition, shape):
            if x is None or not condition:
                return None
            else:
                return torch.tensor(x.reshape(shape)).to(batch)

        output = _OpenMMOutput(
            energies=to_tensor(result.energies, evaluate_energy, (*sample_shape, 1)),
            forces=to_tensor(result.forces, evaluate_force, batch.shape),
            new_positions=to_tensor(result.new_positions, evaluate_positions, batch.shape),
            new_box_vectors=to_tensor(result.new_box_vectors, evaluate_box_vectors, (*sample_shape, 3, 3)),
            new_velocities=to_tensor(result.new_velocities, evaluate_velocities, batch.shape),
            log_path_probability_ratio=(to_tensor(
                result.log_path_probability_ratio,
                evaluate_path_probability_ratio,
                (*sample_shape, 1)
            ))
        )

        # store
        self.last_output = output

        return output


class MultiContext:
    """A container for multiple OpenMM Contexts that are operated by different worker processes.

    Parameters:
    -----------
    n_workers : int
        The number of workers which operate one context each.
    system : openmm.System
        The system that contains all forces.
    integrator : openmm.Integrator
        An OpenMM integrator.
    platform_name : str
        The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
    platform_properties : dict, optional
        A dictionary of platform properties.
    """

    def __init__(self, n_workers, system, integrator, platform_name, platform_properties={}):
        """Set up workers and queues."""
        self._n_workers = n_workers
        self._system = system
        self._integrator = integrator
        self._platform_name = platform_name
        self._platform_properties = platform_properties
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers = []  # workers are initialized in first evaluate call
        # using multiple workers
        try:
            get_ipython
            warnings.warn(
                "It looks like you are using an OpenMMBridge with multiple workers in an ipython environment. "
                "This can behave a bit silly upon KeyboardInterrupt (e.g., kill the stdout stream). "
                "If you experience any issues, consider initializing the bridge with n_workers=1 in ipython/jupyter.",
                UserWarning
            )
        except NameError:
            pass

    def _reinitialize(self):
        """Reinitialize the MultiContext"""
        self.terminate()
        # recreate objects
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers = []
        for i in range(self._n_workers):
            worker = MultiContext.Worker(
                self._task_queue,
                self._result_queue,
                self._system, self._integrator,
                self._platform_name,
                self._platform_properties,
            )
            self._workers.append(worker)
            worker.start()

    def evaluate(self, openmm_input):
        """Delegate energy and force computations to the workers.

        Parameters
        ----------
        openmm_input : _OpenMMInput
            Input namedtuple, where positions are batched numpy arrays of shape (batchsize, n_particles, 3).

        Returns
        -------
        openmm_output : _OpenMMOutput
            Output namedtuple, where forces are batched numpy arrays of shape (batchsize, n_particles, 3).
        """
        if not self.is_alive():
            self._reinitialize()
        openmm_input.validate()

        try:
            # iterate over positions, box_vectors, velocities, temperature_scalings
            for i, item in enumerate(openmm_input.items):
                self._task_queue.put([i, *item])
            results = [self._result_queue.get() for _ in openmm_input.positions]
        except Exception as e:
            self.terminate()
            raise e

        # sort by index and combine into batch
        results = sorted(results, key=lambda x: x[0])
        openmm_output = _OpenMMOutput.collate_batch([_OpenMMOutput(*res[1:]) for res in results])
        return openmm_output

    def is_alive(self):
        """Whether all workers are alive."""
        return all(worker.is_alive() for worker in self._workers) and len(self._workers) > 0

    def terminate(self):
        """Terminate the workers."""
        # soft termination
        for _ in self._workers:
            self._task_queue.put(None)
        # hard termination
        #for worker in self._workers:
        #    worker.terminate()

    def __del__(self):
        self.terminate()

    class Worker(mp.Process):
        """A worker process that computes energies in its own context.

        Parameters:
        -----------
        task_queue : multiprocessing.Queue
            The queue that the MultiContext pushes tasks to.
        result_queue : multiprocessing.Queue
            The queue that the MultiContext receives results from.
        system : openmm.System
            The system that contains all forces.
        integrator : openmm.Integrator
            An OpenMM integrator.
        platform_name : str
            The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
        platform_properties : dict
            A dictionary of platform properties.
        """

        def __init__(self, task_queue, result_queue, system, integrator, platform_name, platform_properties):
            super(MultiContext.Worker, self).__init__()
            self._task_queue = task_queue
            self._result_queue = result_queue
            self._openmm_system = system
            self._openmm_integrator = pickle.loads( pickle.dumps(integrator))
            self._openmm_platform_name = platform_name
            self._openmm_platform_properties = platform_properties
            self._openmm_context = None

        def run(self):
            """Run the process: set positions and compute energies and forces.
            Positions and box vectors are received from the task_queue in units of nanometers.
            Energies and forces are pushed to the result_queue in units of kJ/mole and kJ/mole/nm, respectively.
            """
            try:
                from openmm import unit, Platform, Context
            except ImportError:
                from simtk import unit   # old openmm versions
                from simtk.openmm import Platform, Context

            # create the context
            # it is crucial to do that in the run function and not in the constructor
            # for some reason, the CPU platform hangs if the context is created in the constructor
            # see also https://github.com/openmm/openmm/issues/2602
            openmm_platform = Platform.getPlatformByName(self._openmm_platform_name)
            self._openmm_context = Context(
                self._openmm_system,
                self._openmm_integrator,
                openmm_platform,
                self._openmm_platform_properties
            )
            self._openmm_context.reinitialize(preserveState=True)

            # get tasks from the task queue
            for task in iter(self._task_queue.get, None):
                index, *inputs = task
                openmm_input = _OpenMMInput(*inputs)
                openmm_output = openmm_evaluate(self._openmm_context, openmm_input)
                # push energies and forces to the results queue
                self._result_queue.put(
                    [index, *openmm_output]
                )


class SingleContext:
    """Mimics the MultiContext API but does not spawn worker processes.

    Parameters:
    -----------
    n_workers : int
        Needs to be 1.
    system : openmm.System
        The system that contains all forces.
    integrator : openmm.Integrator
        An OpenMM integrator.
    platform_name : str
        The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
    platform_properties : dict, optional
        A dictionary of platform properties.
    """

    def __init__(self, n_workers, system, integrator, platform_name, platform_properties={}):
        """Set up workers and queues."""
        try:
            from openmm import Platform, Context
        except ImportError:
            from simtk.openmm import Platform, Context   # old openmm versions

        assert n_workers == 1
        openmm_platform = Platform.getPlatformByName(platform_name)
        self._openmm_context = Context(system, integrator, openmm_platform, platform_properties)

    def evaluate(self, openmm_input):
        """Compute energies and/or forces.

        Parameters
        ----------
        openmm_input : _OpenMMInput
            Input namedtuple, where positions are batched numpy arrays of shape (batchsize, n_particles, 3).

        Returns
        -------
        openmm_output : _OpenMMOutput
            Output namedtuple, where forces are batched numpy arrays of shape (batchsize, n_particles, 3).
        """
        openmm_input.validate()
        openmm_outputs = []
        for item in openmm_input.items:
            openmm_outputs.append(openmm_evaluate(self._openmm_context, item))
        return _OpenMMOutput.collate_batch(openmm_outputs)


def openmm_evaluate(context, openmm_input):
    """OpenMM computation.

    Parameters
    ----------
    context : openmm.Context
    openmm_input : _OpenMMInput
        The input namedtuple containing a single state, where the positions are a numpy array of shape (n_particles, 3).


    Returns
    -------
    openmm_output : _OpenMMOutput
        The output namedtuple, where the forces are a numpy array of shape (n_particles, 3).
    """
    try:
        from openmm import unit
    except ImportError:
        from simtk import unit  # old openmm versions

    try:
        # set context state
        context.setPositions(openmm_input.positions)
        if openmm_input.box_vectors is not None:
            context.setPeriodicBoxVectors(*openmm_input.box_vectors)
        if openmm_input.velocities is not None:
            context.setVelocities(openmm_input.velocities)

        # set integrator temperature
        integrator = context.getIntegrator()
        base_temperature = integrator.getTemperature().value_in_unit(unit.kelvin)
        if openmm_input.temperature_scaling is not None:
            integrator.setTemperature(base_temperature * openmm_input.temperature_scaling * unit.kelvin)
        kBT = (integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilojoule_per_mole)

        # propagate
        log_path_probability_ratio = context.getIntegrator().step(openmm_input.n_simulation_steps)

        # reset temperature
        if openmm_input.temperature_scaling is not None:
            integrator.setTemperature(base_temperature * unit.kelvin)

        # compute energy and forces; fetch state
        state = context.getState(
            getEnergy=openmm_input.evaluate_energy,
            getForces=openmm_input.evaluate_force,
            getPositions=openmm_input.evaluate_positions or openmm_input.evaluate_box_vectors,
            getVelocities=openmm_input.evaluate_velocities
        )
        # parse results
        openmm_output = _OpenMMOutput(
            energies=(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                / kBT
                if openmm_input.evaluate_energy else None
            ),
            forces=(
                state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                / kBT
                if openmm_input.evaluate_force else None
            ),
            new_positions=(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
                if openmm_input.evaluate_positions else None
            ),
            new_box_vectors=(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)
                if openmm_input.evaluate_box_vectors else None
            ),
            new_velocities=(
                state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers / unit.picoseconds)
                if openmm_input.evaluate_velocities else None
            ),
            log_path_probability_ratio=log_path_probability_ratio
        )
    except Exception as e:
        if openmm_input.err_handling == "exception":
            raise e
        else:
            openmm_output = _OpenMMOutput(
                energies=numpy.infty,
                forces=numpy.nan * numpy.ones_like(openmm_input.positions),
                new_positions=openmm_input.positions,
                new_box_vectors=openmm_input.box_vectors,
                new_velocities=openmm_input.velocities,
                log_path_probability_ratio=0.0
            )
            if openmm_input.err_handling == "warning":
                warnings.warn("Suppressed exception: {}".format(e))
            elif openmm_input.err_handling == "ignore":
                pass
            else:
                assert False
    return openmm_output


class _OpenMMInput(namedtuple(
    "OpenMMInput",
    (
        "positions", "box_vectors", "velocities", "temperature_scaling", "evaluate_energy", "evaluate_force",
        "evaluate_positions", "evaluate_box_vectors", "evaluate_velocities",
        "evaluate_path_probability_ratio", "err_handling", "n_simulation_steps",
    ), defaults=(None, None, None, True, True, False, False, False, False, "warning", 0)
)):
    """Inputs to the openmm computations.

    Parameters
    ----------
    positions : Union[numpy.ndarray, torch.Tensor]
        The particle positions in nanometer; its shape is (batch_size, num_particles, 3).
    box_vectors :Union[numpy.ndarray, torch.Tensor], optional
        The periodic box vectors in nanometer; its shape is (batch_size, 3, 3).
        If not specified, don't change the box vectors.
    velocities : Union[numpy.ndarray, torch.Tensor], optional
        The velocities in nanometer/picosecond; its shape is (batch_size, num_particles, 3).
    temperature_scaling : Union[numpy.ndarray, torch.Tensor], optional
        Scaling factors to the simulation temperature.
    evaluate_energy : bool, optional
        Whether to compute energies.
    evaluate_force : bool, optional
        Whether to compute forces.
    evaluate_positions : bool, optional
        Whether to return positions.
    evaluate_box_vectors : bool, optional
        Whether to return box vectors.
    evaluate_velocities : bool, optional
        Whether to return velocities.
    evaluate_path_probability_ratio : bool, optional
        Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.
    err_handling : str, optional
        How to handle infinite energies (one of {"warning", "ignore", "exception"}).
    n_simulation_steps : int, optional
        If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.
    """

    @property
    def items(self):
        """iterator over items in the batch"""
        box_vectors = [None for _ in self.positions] if self.box_vectors is None else self.box_vectors
        velocities = [None for _ in self.positions] if self.velocities is None else self.velocities
        temperature_scaling = [None for _ in self.positions] if self.temperature_scaling is None else self.temperature_scaling
        for i in range(len(self.positions)):
            yield _OpenMMInput(self.positions[i], box_vectors[i], velocities[i], temperature_scaling[i], *self[4:])

    def validate(self):
        assert self.box_vectors is None or len(self.box_vectors) == len(self.positions), \
            "box_vectors and positions have to be the same length"
        assert self.velocities is None or self.velocities.shape == self.positions.shape, \
            "velocities and positions have to be the same length"
        assert self.temperature_scaling is None or self.temperature_scaling.shape == (len(self.positions), )


class _OpenMMOutput(namedtuple(
    "OpenMMOutput",
    ("energies", "forces", "new_positions", "new_box_vectors", "new_velocities", "log_path_probability_ratio"),
    defaults=(None, None, None, None, None, None)
)):
    """Outputs from the openmm computations.

    Attributes
    ----------
    energies : Union[torch.Tensor, np.ndarray, float, NoneType]
        The energies in units of kilojoule/mole
    forces : Union[torch.Tensor, np.ndarray, NoneType]
        The forces in units of kilojoule/mole/nm
    new_positions : Union[torch.Tensor, np.ndarray, NoneType]
        The updated positions in units of nm
    new_box_vectors : Union[torch.Tensor, np.ndarray, NoneType]
        The updated box vectors in units of nm
    new_velocities : Union[torch.Tensor, np.ndarray, NoneType]
        The updated velocities in units of nm/ps
    log_path_probability_ratio : Union[torch.Tensor, np.ndarray, float, NoneType]
        The logarithmic path probability ratios
    """
    @staticmethod
    def collate_batch(items):
        """collate multiple outputs into a batch"""
        result = _OpenMMOutput(
            energies=(
                None if any(item.energies is None for item in items)
                else np.stack([item.energies for item in items], axis=0)),
            forces=(
                None if any(item.forces is None for item in items)
                else np.stack([item.forces for item in items], axis=0)),
            new_positions=(
                None if any(item.new_positions is None for item in items)
                else np.stack([item.new_positions for item in items], axis=0)),
            new_box_vectors=(
                None if any(item.new_box_vectors is None for item in items)
                else np.stack([item.new_box_vectors for item in items], axis=0)),
            new_velocities=(
                None if any(item.new_velocities is None for item in items)
                else np.stack([item.new_velocities for item in items], axis=0)),
            log_path_probability_ratio=(
                None if any(item.log_path_probability_ratio is None for item in items)
                else np.array([item.log_path_probability_ratio for item in items]))
        )
        return result
