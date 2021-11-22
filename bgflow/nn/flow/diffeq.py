import torch

from .base import Flow
from .dynamics import (
    DensityDynamics,
    InversedDynamics,
    AnodeDynamics
)


class DiffEqFlow(Flow):
    """
    Neural Ordinary Differential Equations flow :footcite:`chen2018neural`
    with the choice of optimize than discretize (use_checkpoints=False)
    and discretize than optimize :footcite:`gholami2019anode` (use_checkpoints=True) for the ODE solver.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
            self,
            dynamics,
            integrator="dopri5",
            atol=1e-10,
            rtol=1e-5,
            n_time_steps=2,
            t_max=1.,
            use_checkpoints=False,
            **kwargs
    ):
        super().__init__()
        self._dynamics = DensityDynamics(dynamics)
        self._inverse_dynamics = DensityDynamics(InversedDynamics(dynamics, t_max))
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps
        self._t_max = t_max
        self._use_checkpoints = use_checkpoints
        self._kwargs = kwargs

    def _forward(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._dynamics, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._inverse_dynamics, **kwargs)

    def _run_ode(self, *xs, dynamics, **kwargs):
        """
        Runs the ODE solver.

        Parameters
        ----------
        xs : PyTorch tensor
            The current configuration of the system
        dynamics : PyTorch module
            A dynamics function that computes the change of the system and its density.

        Returns
        -------
        ys : PyTorch tensor
            The new configuration of the system after being propagated by the dynamics.
        dlogp : PyTorch tensor
            The change in log density due to the dynamics.
        """

        assert (all(x.shape[0] == xs[0].shape[0] for x in xs[1:]))
        n_batch = xs[0].shape[0]
        logp_init = torch.zeros(n_batch, 1).to(xs[0])
        state = (*xs, logp_init)
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(xs[0])
        kwargs = {**self._kwargs, **kwargs}
        if not self._use_checkpoints:
            from torchdiffeq import odeint_adjoint
            *ys, dlogp = odeint_adjoint(
                dynamics,
                state,
                t=ts,
                method=self._integrator_method,
                rtol=self._integrator_rtol,
                atol=self._integrator_atol,
                options=kwargs
            )
            ys = [y[-1] for y in ys]
        else:
            from anode.adjoint import odesolver_adjoint
            state = torch.cat(state, dim=-1)
            anode_dynamics = AnodeDynamics(dynamics)
            state = odesolver_adjoint(anode_dynamics, state, options=kwargs)
            ys = [state[:, :-1]]
            dlogp = [state[:, -1:]]
        dlogp = dlogp[-1]
        return (*ys, dlogp)
