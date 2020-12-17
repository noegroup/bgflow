import torch


class InversedDynamics(torch.nn.Module):
    """
    Inverse of a dynamics for the inverse flow.
    """

    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def forward(self, t, state):
        """
        Evaluates the change of the system `dxs` at time `t_max` - `t` for the inverse dynamics.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        state : PyTorch tensor
            The current state of the system

        Returns
        -------
        [-*dxs, -dlogp] : List of PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`-dxs`) and the update of the log density (`-dlogp`).
        """

        *dxs, dlogp = self._dynamics(self._t_max - t, state)
        return [-dx for dx in dxs] + [-dlogp]
