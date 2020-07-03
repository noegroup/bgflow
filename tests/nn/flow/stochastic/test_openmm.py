

from bgtorch.nn.flow.stochastic.openmm import BrownianPathProbabilityIntegrator

import pytest
import numpy as np
from simtk import openmm as mm
from simtk import unit
import copy


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


@pytest.mark.parametrize("integrator", [BrownianPathProbabilityIntegrator(300*unit.kelvin, 1, 0.0001)])
def test_path_probability(integrator):
    """
    We check the SNF as follows: Setup a harmonic oscillator. Initialize particles according to a
    Gaussian that is broader or narrower than the oscillator and make sure that the sign of the
    log determinant of the jacobian tracks the contraction/expansion during equilibration.
    """
    n_steps = 100
    n_particles = 1000
    force_constant = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2

    system = make_harmonic_system(n_particles, force_constant)

    kT = integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
    base_sigma = np.sqrt((kT/force_constant))
    # the standard deviation that corresponds to the force constant and temperature

    ratios = {}
    for sigma, name in [(base_sigma, "base"), (2*base_sigma, "broad"), (0.5*base_sigma, "narrow")]:
        positions = 0.5*unit.nanometer + sigma*np.random.randn(n_particles, 3)
        integ = copy.deepcopy(integrator)
        context = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(positions)
        ratios[name] = integ.step(n_steps) / n_particles

    assert ratios["broad"] < 0  # broad distribution should narrow
    assert ratios["narrow"] > 0  # narrow distribution should broaden
    assert ratios["base"] > ratios["broad"]
    assert ratios["base"] < ratios["narrow"]
    assert abs(ratios["base"]) < 1e-1  # correct distribution should have a log path density ratio of approximately 0

    # check if we can bind the reverse integrator to a context and use it
    mm.Context(system, integ.get_reverse_integrator(), mm.Platform.getPlatformByName("Reference"))
    assert integ.get_reverse_integrator().ratio == 0.0


@pytest.mark.parametrize(
    "integrator,reference_integrator",
    [
        (BrownianPathProbabilityIntegrator(300, 1, 0.001), mm.BrownianIntegrator(300, 1, 0.001))
    ]
)
def test_temperature(integrator, reference_integrator):
    n_particles = 100
    n_steps = 1000
    force_constant = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2
    system = make_harmonic_system(n_particles, force_constant)

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