import pytest
import torch
import warnings

import numpy as np

from bgtorch.distribution import Energy
from bgtorch.distribution.energy.openmm import OpenMMBridge, OpenMMEnergy


class OneParticleTestBridge(OpenMMBridge):
    """OpenMM bridge for a system with one particle"""

    def __init__(self, n_workers=1, n_simulation_steps=0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", DeprecationWarning
                )  # ignore warnings inside OpenMM
                from simtk import openmm, unit
        except ImportError:
            pytest.skip("Test requires OpenMM.")

        # a system with one particle and an external dummy force
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)
        force = openmm.CustomExternalForce("x")
        force.addParticle(0)
        system.addForce(force)

        super(OneParticleTestBridge, self).__init__(
            system,
            openmm.LangevinIntegrator(
                300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds
            ),
            n_workers=n_workers,
            n_simulation_steps=n_simulation_steps,
        )


class DummyEnergy(Energy):
    def __init__(self, dim):
        super().__init__(dim=dim)

    def _energy(self, *xs, **kwargs):
        return sum(
            0.5 * x.pow(2).view(x.shape[0], -1).sum(dim=-1, keepdim=True) for x in xs
        )


def test_energy_event_parser():
    from bgtorch.distribution.energy.base import (
        _is_non_empty_sequence_of_integers,
        _is_sequence_of_non_empty_sequences_of_integers,
    )

    assert _is_non_empty_sequence_of_integers([10, 2, 6, 9])
    assert _is_non_empty_sequence_of_integers(torch.Size([10, 20, 3]))
    assert not _is_non_empty_sequence_of_integers([10, 2, 2.4])
    assert not _is_non_empty_sequence_of_integers([10, 2, [3]])
    assert not _is_non_empty_sequence_of_integers([[10], [10]])

    assert _is_sequence_of_non_empty_sequences_of_integers([[10, 10], [5], [10]])
    assert _is_sequence_of_non_empty_sequences_of_integers(
        [torch.Size([10, 10]), torch.Size([10])]
    )
    assert _is_sequence_of_non_empty_sequences_of_integers(
        [[10, 5], torch.Size([10, 10])]
    )
    assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 10], 10, [10]])
    assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 5], [10, 2.0]])
    assert not _is_sequence_of_non_empty_sequences_of_integers([10, 10])
    assert not _is_sequence_of_non_empty_sequences_of_integers([[], [10]])
    assert not _is_sequence_of_non_empty_sequences_of_integers(torch.Size([10, 10]))
    assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 10, 10]])


@pytest.mark.parametrize("batch", [[23], [23, 71], [23, 71, 13]])
def test_energy_event_types(batch):

    envs = [torch.no_grad, torch.enable_grad]
    for env in envs:
        with env():

            # test single dimension input
            dim = 11
            dummy = DummyEnergy(dim)
            x = torch.randn(*batch, dim)
            f = dummy.force(x)
            assert torch.allclose(-x, f)
            assert dummy.dim == 11
            assert dummy.event_shape == torch.Size([11])

            # this should fail (too many inputs)
            failed = False
            try:
                dummy.force(x, x)
            except AssertionError:
                failed = True
            assert failed

            # test tensor input
            shape = [11, 7, 4, 3]
            dummy = DummyEnergy(shape)
            x = torch.randn(*batch, *shape)
            f = dummy.force(x)
            assert torch.allclose(-x, f)
            assert dummy.dim == 11 * 7 * 4 * 3
            assert dummy.event_shape == torch.Size([11, 7, 4, 3])

            # this should fail (too many inputs)
            failed = False
            try:
                dummy.force(x, x)
            except AssertionError:
                failed = True
            assert failed

            # test multi-tensor input
            shapes = [[11, 7], [5, 3], [13, 17]]
            dummy = DummyEnergy(shapes)
            x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
            fx, fy, fz = dummy.force(x, y, z)
            assert all(torch.allclose(-x, f) for (x, f) in zip([x, y, z], [fx, fy, fz]))
            fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
            assert fy is None and all(
                torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
            )

            # this should fail: inconsistent batch dimension
            failed = False
            try:
                batches = [[5, 7], [5, 7], [5, 6]]
                x, y, z = [
                    torch.randn(*batch, *shape)
                    for (batch, shape) in zip(batches, shapes)
                ]
                fx, fy, fz = dummy.force(x, y, z)
            except AssertionError:
                failed = True
            assert failed

            # this should fail: wrong input shapes
            failed = False
            try:
                batches = [[5, 7], [5, 7], [5, 7]]
                x, y, z = [
                    torch.randn(*batch, *shape)
                    for (batch, shape) in zip(batches, shapes)
                ]
                y = y[..., :-1]
                fx, fy, fz = dummy.force(x, y, z)
            except AssertionError:
                failed = True
            assert failed

            # this should fail: dim not defined for multiple tensor input
            failed = False
            try:
                dummy.dim
            except ValueError:
                failed = True
            assert failed

            # this should fail: single event_shape is not defined for multipe tensor input
            failed = False
            try:
                dummy.event_shape
            except ValueError:
                failed = True
            assert failed

            # this should fail (too few inputs)
            failed = False
            try:
                dummy.force(x, y)
            except AssertionError:
                failed = True
            assert failed

            # this should fail (too many inputs)
            failed = False
            try:
                dummy.force(x, y, z, x)
            except AssertionError:
                failed = True
            assert failed

            # test that `requires_grad` state of input vars stays preserved
            shapes = [[11, 7], [5, 3], [13, 17]]
            dummy = DummyEnergy(shapes)
            x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
            x.requires_grad_(True)
            y.requires_grad_(False)
            z.requires_grad_(False)
            fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
            assert x.requires_grad and not (y.requires_grad) and (not z.requires_grad)
            assert fy is None and all(
                torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
            )

            # test for singular shapes in multi-tensor setting
            shapes = [[11], [5], [13]]
            dummy = DummyEnergy(shapes)
            x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
            fx, fy, fz = dummy.force(x, y, z)
            assert all(torch.allclose(-x, f) for (x, f) in zip([x, y, z], [fx, fy, fz]))
            fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
            assert fy is None and all(
                torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
            )


@pytest.mark.parametrize("n_workers", [1, 2, 8])
@pytest.mark.parametrize("n_simulation_steps", [0, 100])
def test_openmm_bridge_evaluate_dummy(n_workers, n_simulation_steps):
    """Test if we can evaluate an energy; skip test if openmm is not installed."""

    bridge = OneParticleTestBridge(n_workers, n_simulation_steps)

    from simtk import unit

    # test forces and energies generated by the bridge
    batch_size = 4
    positions = torch.tensor([[0.1, 0.0, 0.0]] * batch_size)
    kT = unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin
    energies, forces, *_ = bridge.evaluate(positions)
    assert energies.shape == torch.Size([batch_size, 1])
    assert forces.shape == torch.Size([batch_size, 3])
    if n_simulation_steps == 0:
        assert energies.numpy()[0] == pytest.approx(
            [0.1 * unit.kilojoule_per_mole / kT], abs=1e-8, rel=0
        )
        assert forces.numpy()[0] == pytest.approx(
            [-1.0 * unit.kilojoule_per_mole / kT, 0.0, 0.0], abs=1e-8, rel=0
        )
    else:
        assert energies.numpy()[0] != pytest.approx(
            [0.1 * unit.kilojoule_per_mole / kT], abs=1e-8, rel=0
        )


# run 'pytest --durations 0' to print the time it takes to evaluate
# change batch_size and n_workers to profile
@pytest.mark.parametrize("testsystem_name", ["AlanineDipeptideImplicit", "WaterBox"])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("n_workers", [1, 8])
def test_openmm_bridge_evaluate_openmmtools_testsystem(
    testsystem_name, batch_size, n_workers
):
    """Test if we can evaluate an energy; skip test if openmm is not installed."""
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="numpy.ufunc size changed"
    )

    if testsystem_name == "WaterBox" and n_workers == 1:
        pytest.skip()
    # prevent openmm hanging

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", DeprecationWarning
            )  # ignore warnings inside OpenMM
            from simtk import openmm, unit
            from openmmtools import testsystems

            Testsystem = getattr(testsystems, testsystem_name)
    except ImportError:
        pytest.skip("Test requires OpenMM and openmmtools.")

    # alanine dipeptide testsystem
    testsystem = Testsystem()
    system = testsystem.system

    # test forces and energies generated by the bridge
    bridge = OpenMMBridge(
        system,
        openmm.LangevinIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, 1.0 * unit.femtoseconds
        ),
        n_workers=n_workers,
    )
    batch = torch.tensor(
        np.array(
            [np.ravel(testsystem.positions.value_in_unit(unit.nanometer))] * batch_size
        )
    )
    energies, forces, *_ = bridge.evaluate(batch)
    assert energies.shape == torch.Size([batch_size, 1])
    assert forces.shape == torch.Size([batch_size, 3 * len(testsystem.positions)])


def test_openmm_bridge_cache():
    """Test if hashing and caching works."""
    bridge = OneParticleTestBridge()
    omm_energy = OpenMMEnergy(3, bridge)
    omm_energy._energy(torch.tensor([[0.1, 0.0, 0.0]] * 2))
    hash1 = omm_energy._last_batch
    omm_energy._energy(torch.tensor([[0.2, 0.0, 0.0]] * 2))
    assert omm_energy._last_batch != hash1
    omm_energy._energy(torch.tensor([[0.1, 0.0, 0.0]] * 2))
    assert omm_energy._last_batch == hash1
    omm_energy._energy(torch.tensor([[0.1, 0.0, 0.0]] * 2))

    # test if forces are in the same memory location for same input batch
    force_address = hex(id(omm_energy._openmm_energy_bridge.last_forces))
    force = (
        omm_energy._openmm_energy_bridge.last_forces
    )  # retain a pointer to last forces so that memory is not freed
    assert (
        hex(id(omm_energy.force(torch.tensor([[0.1, 0.0, 0.0]] * 2)))) == force_address
    )
    assert (
        hex(id(omm_energy.force(torch.tensor([[0.1, 0.0, 0.0]] * 2)))) == force_address
    )
    assert (
        hex(id(omm_energy.force(torch.tensor([[0.2, 0.0, 0.0]] * 2)))) != force_address
    )
    assert (
        hex(id(omm_energy.force(torch.tensor([[0.1, 0.0, 0.0]] * 2)))) != force_address
    )

    # suppress flake8 F841 warning
    force


# Returned log path probability ratios / positions are tested in the SNF tests
