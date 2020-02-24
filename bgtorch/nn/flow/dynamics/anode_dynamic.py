import torch

class AnodeDynamics(torch.nn.Module):
    """
    Converts the the concatenated state, which is required for the ANODE ode solver,
    to the tuple (`xs`, `dlogp`) for the following dynamics functions.
    Parameters
    ----------
    t: PyTorch tensor
        The current time
    state: PyTorch tensor
        The current state of the system
    Returns
    -------
    state: PyTorch tensor
        The combined state update of shape `[n_batch, n_dimensions]`
        containing the state update of the system state `dx/dt`
        (`state[:, :-1]`) and the update of the log density (`state[:, -1]`).
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics

    def forward(self, t, state):
        xs = state[:, :-1]
        dlogp = state[:, -1:]
        state = (xs, dlogp)
        *dxs, div = self._dynamics(t, state)
        state = torch.cat([*dxs, div], dim=-1)
        return state
