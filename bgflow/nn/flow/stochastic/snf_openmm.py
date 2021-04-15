"""
OpenMM Interface for Stochastic Normalizing Flows
"""

import pickle
import warnings

from ..base import Flow

try:
    from openmmtools.integrators import ThermostatedIntegrator
except ImportError:

    class ThermostatedIntegrator:
        def __init__(self, *args, **kwargs):
            raise ImportError("openmmtools not installed")


__all__ = [
    "OpenMMStochasticFlow",
    "PathProbabilityIntegrator",
    "BrownianPathProbabilityIntegrator",
]


class OpenMMStochasticFlow(Flow):
    """A stochastic normalizing flow that performs integration steps in OpenMM and computes the Jacobian determinant
    on the way.

    Parameters
    ----------
    openmm_bridge : bgflow.distribution.energy.openmm.OpenMMBridge
        The openmm energy bridge instance that propagates the SNF.

    inverse_openmm_bridge : None or bgflow.distribution.energy.openmm.OpenMMBridge
        The bridge instance that propagates the inverse transform. If None, use the forward transform.

    Examples
    --------
    The class should be used with one of the integrators defined in this module.
    For example
    >>>     from bgflow.distribution.energy.openmm import OpenMMBridge
    >>>     from openmmtools.testsystems import AlanineDipeptideImplicit
    >>>     integrator = BrownianPathProbabilityIntegrator(1, 100, 0.001)
    >>>     ala2 = AlanineDipeptideImplicit()
    >>>     bridge = OpenMMBridge(ala2.system, integrator, n_workers=1, n_simulation_steps=5)
    >>>     snf = OpenMMStochasticFlow(bridge)

    >>>     x = torch.tensor(ala2.positions.value_in_unit(unit.nanometer)).view(1,len(ala2.positions)*3)
    >>>     y, dlogP = snf._forward(x)
    """

    def __init__(self, openmm_bridge, inverse_openmm_bridge=None):
        super(OpenMMStochasticFlow, self).__init__()
        self.openmm_bridge = self._check_bridge(openmm_bridge)
        if inverse_openmm_bridge is not None:
            self.inverse_openmm_bridge = self._check_bridge(inverse_openmm_bridge)
        else:
            self.inverse_openmm_bridge = openmm_bridge

    def _forward(self, *xs, **kwargs):
        _, _, y, dlog = self.openmm_bridge.evaluate(
            xs[0],
            evaluate_force=False,
            evaluate_energy=False,
            evaluate_positions=True,
            evaluate_path_probability_ratio=True,
        )
        return y, dlog

    def _inverse(self, *xs, **kwargs):
        _, _, y, dlog = self.inverse_openmm_bridge.evaluate(
            xs[0],
            evaluate_force=False,
            evaluate_energy=False,
            evaluate_positions=True,
            evaluate_path_probability_ratio=True,
        )
        return y, dlog

    @staticmethod
    def _check_bridge(bridge):
        assert isinstance(
            bridge.integrator, PathProbabilityIntegrator
        ), "OpenMMStochasticFlow requires an integrator that tracks the log path probability ratio."
        assert (
            bridge.n_simulation_steps > 0
        ), "OpenMMStochasticFlow requires a bridge that performs integration steps in OpenMM."
        return bridge


class PathProbabilityIntegrator(ThermostatedIntegrator):
    """Abstract base class for path probability integrators.
    These integrators track the path probability ratio which is required in stochastic normalizing flows.

    Parameters
    ----------
    temperature : float or unit.Quantity
        Temperature in kelvin.
    stepsize : float or unit.Quantity
        Step size in picoseconds.

    Attributes
    ----------
    ratio : float
        The logarithmic path probability ratio summed over all steps taken during the previous invocation of `step`.
    """

    def __init__(self, temperature, stepsize):
        super(PathProbabilityIntegrator, self).__init__(temperature, stepsize)
        self.addGlobalVariable("log_path_probability_ratio", 0.0)

    @property
    def ratio(self):
        return self.getGlobalVariableByName("log_path_probability_ratio")

    @ratio.setter
    def ratio(self, value):
        self.setGlobalVariableByName("log_path_probability_ratio", value)

    def step(self, n_steps):
        """Propagate the system using the integrator.
        This method returns the current log path probability ratio and resets it to 0.0 afterwards.

        Parameters
        ----------
        n_steps : int
            The number of steps

        Returns
        -------
        ratio : float
            The logarithmic path probability ratio summed over n_steps steps.
        """
        self.ratio = 0.0
        super().step(n_steps)
        ratio = self.ratio
        return ratio

    def get_reverse_integrator(self):
        """Default behavior: the forward and reverse integrator are identical."""
        pickled = pickle.dumps(self)
        reverse = pickle.loads(pickled)
        reverse.ratio = 0
        return reverse


class BrownianPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Overdamped Langevin Dynamics"""

    def __init__(self, temperature, friction_coeff, stepsize):
        super(BrownianPathProbabilityIntegrator, self).__init__(temperature, stepsize)
        warnings.warn(
            "The current implementation of the BrownianPathProbabilityIntegrator "
            "does not support force derivatives. Therefore, the gradients of the log path probability ratio "
            "will be inaccurate.",
            UserWarning,
        )

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
        self.addComputePerDof(
            "x", "x+epsilon*f + sqrt(2*epsilon*kT)*w"
        )  # position update
        self.addComputePerDof(
            "w_", "sqrt(epsilon/2/kT) * (- f_old - f) - w"
        )  # backward noise
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-x_old)/dt")
        self.addConstrainVelocities()

        # update logarithmic path probability ratio
        self.addComputeSum("wsquare", "w*w")
        self.addComputeSum("w_square", "w_*w_")
        self.addComputeGlobal(
            "log_path_probability_ratio",
            "log_path_probability_ratio-0.5*(w_square - wsquare)",
        )


class LangevinPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Langevin Dynamics"""

    def __init__(self, temperature, friction_coeff, stepsize):
        super(LangevinPathProbabilityIntegrator, self).__init__(temperature, stepsize)
        raise NotImplementedError()


class MCMCPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Markov Chain Monte Carlo"""

    def __init__(self, temperature, friction_coeff, stepsize):
        super(MCMCPathProbabilityIntegrator, self).__init__(temperature, stepsize)
        raise NotImplementedError()


class HamiltonianMCPathProbabilityIntegrator(PathProbabilityIntegrator):
    """Hamiltonial Monte Carlo Integrator"""

    def __init__(self, temperature, friction_coeff, stepsize):
        super(HamiltonianMCPathProbabilityIntegrator, self).__init__(
            temperature, stepsize
        )
        raise NotImplementedError()
