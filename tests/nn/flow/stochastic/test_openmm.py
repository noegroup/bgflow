

from bgtorch.nn.flow.stochastic.openmm import BrownianPathProbabilityIntegrator

import pytest
import numpy as np
from simtk import openmm as mm
from simtk import unit
import copy


@pytest.mark.parametrize("integrator", [BrownianPathProbabilityIntegrator(300*unit.kelvin, 1, 0.0001)])
def test_path_probability_integrator(integrator):
    """
    We check the SNF as follows: Setup a harmonic oscillator. Initialize particles according to a
    Gaussian that is broader or narrower than the oscillator and make sure that the sign of the
    log determinant of the jacobian tracks the contraction/expansion during equilibration.
    """

    n_particles = 1000
    n_steps = 100
    temperature = integrator.getTemperature()

    system = mm.System()
    harmonic = mm.CustomExternalForce("0.5 * force_constant * (x-0.5)^2")
    force_constant = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2
    harmonic.addGlobalParameter("force_constant", force_constant)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(1,0,0), mm.Vec3(0,1,0), mm.Vec3(0,0,1))
    for i in range(n_particles):
        system.addParticle(1.0)
        harmonic.addParticle(i)
    system.addForce(harmonic)

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
    assert integ.get_reverse_integrator()._ratio == 0.0