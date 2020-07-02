"""
OpenMM Interface for Stochastic Normalizing Flows
"""

import pickle

from ..base import Flow
from simtk.openmm import CustomIntegrator, XmlSerializer
from openmmtools.integrators import ThermostatedIntegrator


class OpenMMStochasticFlow(Flow):
    """A stochastic normalizing flow that performs integration steps in OpenMM and computes the Jacobian determinant
    on the way.

    Parameters
    ----------
    openmm_system : simtk.openmm.System
        The system object that defines molecular energies and forces.

    stochastic_flow_integrator : simtk.openmm.CustomIntegrator
        A custom integrator that has a getter for the path probability ratio.

    reverse_integrator : simtk.openmm.CustomIntegrator
        The integrator that defines the reverse step. If None, create the reverse integrator
        from the forward integrator.

    """
    def __init__(self, openmm_system, stochastic_flow_integrator, reverse_integrator=None):
        super(OpenMMStochasticFlow, self).__init__()

    def _forward(self, *xs, **kwargs):
        pass

    def _inverse(self, *xs, **kwargs):
        pass


class PathProbabilityIntegrator(ThermostatedIntegrator):
    """Abstract base class for path probability integrators.
    These integrators track the path probability ratio which is required in stochastic normalizing flows.
    """
    def __init__(self, temperature, stepsize):
        super(PathProbabilityIntegrator, self).__init__(temperature, stepsize)
        self.addGlobalVariable("log_path_probability_ratio", 0.0)

    @property
    def ratio(self):
        return self.getGlobalVariableByName("log_path_probability_ratio")

    @ratio.setter
    def ratio(self, value):
        return self.setGlobalVariableByName("log_path_probability_ratio", value)

    def step(self, n_steps):
        super().step(n_steps)
        ratio = self.ratio
        self.ratio = 0.0
        return ratio

    def get_reverse_integrator(self):
        """Default behavior: the forward and reverse integrator are identical."""
        pickled = pickle.dumps(self)
        return pickle.loads(pickled)


class BrownianPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Overdamped Langevin Dynamics"""
    def __init__(self, temperature, friction_coeff, stepsize):
        super(BrownianPathProbabilityIntegrator, self).__init__(temperature, stepsize)

        # variable definitions
        self.addGlobalVariable("gamma", friction_coeff)
        self.addPerDofVariable("w", 0)
        self.addPerDofVariable("w_", 0)
        self.addPerDofVariable("epsilon", 0)
        self.addPerDofVariable("f_old", 0)
        self.addPerDofVariable("x_old", 0)

        # propagation
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"epsilon": "dt/gamma/m"})
        self.addComputePerDof("w", "gaussian")
        self.addComputePerDof("f_old", "f")
        self.addComputePerDof("x_old", "x")
        self.addComputePerDof("x", "x+epsilon*f + sqrt(2*epsilon*kT)*w")  # position update
        self.addComputePerDof("w_", "sqrt(epsilon/2/kT) * (- f_old - f) - w")  # backward noise
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-x_old)/dt")
        self.addConstrainVelocities()

        # update logarithmic path probability ratio
        self.addComputeSum("wsquare", "w*w")
        self.addComputeSum("w_square", "w_*w_")
        self.addComputeGlobal("log_path_probability_ratio", "log_path_probability_ratio-0.5*(w_square - wsquare)")


class LangevinPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Langevin Dynamics"""
    def __init__(self, temperature, friction_coeff, stepsize):
        super(LangevinPathProbabilityIntegrator, self).__init__(temperature, stepsize)


class MCMCPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Markov Chain Monte Carlo"""
    def __init__(self, temperature, friction_coeff, stepsize):
        super(MCMCPathProbabilityIntegrator, self).__init__(temperature, stepsize)


class HamiltonianMCPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Hamiltonial Monte Carlo Integrator"""
    def __init__(self, temperature, friction_coeff, stepsize):
        super(HamiltonianMCPathProbabilityIntegrator, self).__init__(temperature, stepsize)

