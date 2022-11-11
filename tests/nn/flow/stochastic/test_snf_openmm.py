

from bgflow.nn.flow.stochastic.snf_openmm import BrownianPathProbabilityIntegrator, OpenMMStochasticFlow

import copy
import pickle

import pytest
import numpy as np
import torch

try:
    from simtk import openmm as mm
    from simtk import unit
    from openmmtools.integrators import ThermostatedIntegrator
    _OPENMM_INSTALLED = True
except ImportError:
    _OPENMM_INSTALLED = False

pytestmark = [
    pytest.mark.skipif(not _OPENMM_INSTALLED, reason="requires openmm and openmmtools"),
    pytest.mark.filterwarnings("ignore:The current implementation of the BrownianPathProbabilityIntegrator")
]


def _copy_integrator(integrator):
    """Get an identical integrator that is not bound to a Context."""
    return pickle.loads(pickle.dumps(integrator))


def make_harmonic_system(n_particles, force_constant):
    """System with one harmonic oscillator."""
    system = mm.System()
    harmonic = mm.CustomExternalForce("0.5 * force_constant * (x-0.5)^2")
    harmonic.addGlobalParameter("force_constant", force_constant)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(1,0,0), mm.Vec3(0,1,0), mm.Vec3(0,0,1))
    for i in range(n_particles):
        system.addParticle(1.0)
        harmonic.addParticle(i)
    system.addForce(harmonic)
    return system


@pytest.mark.parametrize("IntegratorClass", [BrownianPathProbabilityIntegrator])
def test_path_probability(IntegratorClass):
    """
    We check the SNF as follows: Setup a harmonic oscillator. Initialize particles according to a
    Gaussian that is broader or narrower than the oscillator and make sure that the sign of the
    log determinant of the jacobian tracks the contraction/expansion during equilibration.
    """
    n_steps = 100
    n_particles = 1000
    force_constant = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2

    system = make_harmonic_system(n_particles, force_constant)

    integrator = IntegratorClass(300*unit.kelvin, 1, 0.0001)
    kT = integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
    base_sigma = np.sqrt((kT/force_constant))
    # the standard deviation that corresponds to the force constant and temperature

    ratios = {}
    for sigma, name in [(base_sigma, "base"), (2*base_sigma, "broad"), (0.5*base_sigma, "narrow")]:
        positions = 0.5*unit.nanometer + sigma*np.random.randn(n_particles, 3)
        integ = _copy_integrator(integrator)
        context = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(positions)
        ratios[name] = integ.step(n_steps) / n_particles

    assert ratios["broad"] < 0  # broad distribution should narrow
    assert ratios["narrow"] > 0  # narrow distribution should broaden
    assert ratios["base"] > ratios["broad"]
    assert ratios["base"] < ratios["narrow"]
    assert abs(ratios["base"]) < 0.5  # correct distribution should have a log path density ratio of approximately 0

    # check if we can bind the reverse integrator to a context and use it
    mm.Context(system, integ.get_reverse_integrator(), mm.Platform.getPlatformByName("Reference"))
    assert integ.get_reverse_integrator().ratio == 0.0


@pytest.mark.parametrize("temperature", (1, 500))
@pytest.mark.parametrize("n_workers", (1, 2))
def test_flow_bridge(temperature, n_workers):
    """Test the API of the flow interface"""
    from bgflow.distribution.energy.openmm import OpenMMBridge
    from openmmtools.testsystems import AlanineDipeptideImplicit
    integrator = BrownianPathProbabilityIntegrator(temperature, 100, 0.001)
    ala2 = AlanineDipeptideImplicit()
    bridge = OpenMMBridge(ala2.system, integrator, n_workers=n_workers, n_simulation_steps=4)
    snf = OpenMMStochasticFlow(bridge)

    batch_size = 2
    x = torch.tensor(np.array([ala2.positions.value_in_unit(unit.nanometer)] * batch_size)).view(batch_size,len(ala2.positions)*3)
    y, dlogP = snf._forward(x)
    assert not torch.all(x.isclose(y, atol=1e-3))  # assert that output differs from input
    assert torch.all(x.isclose(y, atol=0.5))  # assert that output does not differ too much
    if temperature < 300:
        assert dlogP[0].item() < 0.0   # should be a space-contracting update since temperature is low
    else:
        assert dlogP[0].item() > 0.0   # should be a space-expanding update since temperature is high


@pytest.mark.parametrize(
    "IntegratorClass", [BrownianPathProbabilityIntegrator]
)
def test_temperature(IntegratorClass):
    ReferenceIntegratorClass = {
        BrownianPathProbabilityIntegrator: mm.BrownianIntegrator
    }[IntegratorClass]
    n_particles = 100
    n_steps = 1000
    force_constant = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2
    system = make_harmonic_system(n_particles, force_constant)

    integrator = IntegratorClass(300, 1, 0.001)
    # copy integrators to enable test repeats
    integrator = _copy_integrator(integrator)
    reference_integrator = ReferenceIntegratorClass(300, 1, 0.001)
    reference_integrator = _copy_integrator(reference_integrator)

    kT = integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
    base_sigma = np.sqrt((kT/force_constant))
    positions = 0.5*unit.nanometer + base_sigma*np.random.randn(n_particles, 3)

    pe = {}
    ke = {}
    x_moments = {}
    v_moments = {}
    for integ, name in [(integrator, "here"), (reference_integrator, "ref")]:
        context = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(positions)
        pe[name] = 0.0
        ke[name] = 0.0
        x_moments[name] = [0.0 for _ in range(6)]
        v_moments[name] = [0.0 for _ in range(6)]
        for _ in range(n_steps):
            integ.step(1)
            state = context.getState(getEnergy=True, getPositions=True, getVelocities=True)
            pe[name] += state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / (n_particles * n_steps)
            ke[name] += state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / (n_particles * n_steps)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            for i in range(6):
                x_moments[name][i] += (pos[:,0]**i).sum() / (n_steps * n_particles)
                v_moments[name][i] += (vel[:,0]**i).sum() / (n_steps * n_particles)
    assert np.isclose(pe["here"], pe["ref"], atol=0.1)                            # potential
    assert np.isclose(ke["here"], ke["ref"], atol=0.1)                            # kinetic
    assert np.isclose(x_moments["here"], x_moments["ref"], atol=0.1).all()        # distribution of positions
    assert np.isclose(v_moments["here"][1], v_moments["ref"][1], atol=0.1).all()  # average velocities ~ 0
    assert np.isclose(v_moments["here"][2], v_moments["ref"][2], rtol=0.2).all()  # average KE/particle
