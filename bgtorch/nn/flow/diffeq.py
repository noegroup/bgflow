import torch

from .base import Flow
from .dynamics import (
    DensityDynamics,
    InversedDynamics
)

#TODO: write docstrings


class DiffEqFlow(Flow):
    def __init__(
        self,
        dynamics,
        integrator="dopri5",
        atol=1e-10,
        rtol=1e-5,
        n_time_steps=2,
        t_max = 1.,
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
        
    def _forward(self, x, **kwargs):
        return self._run_ode(x, self._dynamics, **kwargs)
    
    def _inverse(self, x, **kwargs):
        return self._run_ode(x, self._inverse_dynamics, **kwargs)

    def _run_ode(self, x, dynamics, **kwargs):
        # TODO: kwargs should be parsed to avoid conflicts!
        n_batch = x.shape[0]
        logp_init = torch.zeros(n_batch, 1).to(x)
        state = torch.cat([x, logp_init], dim=-1).contiguous()
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(x)
        kwargs = {**self._kwargs, **kwargs}
        if not self._use_checkpoints:
            from torchdiffeq import odeint_adjoint
            state = odeint_adjoint(
                dynamics,
                state,
                t=ts,
                method=self._integrator_method,
                rtol=self._integrator_rtol,
                atol=self._integrator_atol,
                options=kwargs
            )
        else:
            from anode.adjoint import odesolver_adjoint
            state = odesolver_adjoint(dynamics, state, options=kwargs)
        x = state[-1, :, :-1]
        dlogp = state[-1, :, -1:]
        return x, dlogp
    
    