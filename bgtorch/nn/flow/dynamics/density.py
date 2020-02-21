import torch


class DensityDynamics(torch.nn.Module):
    """
    Computes the change of the system `dx/dt` over at state `x` and
    time `t`. Furthermore, computes the change of density, happening
    due to moving `x` infinitesimally in the direction `dx/dt`
    according to the "instantaneous change of variables rule" [1]
        `dlog(p(x(t))/dt = -div(dx(t)/dt)`
    [1] Neural Ordinary Differential Equations, Chen et. al,
        https://arxiv.org/abs/1806.07366
    Parameters
    ----------
    t: PyTorch tensor
        The current time
    x: PyTorch tensor
        The current state of the system
    Returns
    -------
    dstate: PyTorch tensor
        The combined state update of shape `[n_batch, n_dimensions]`
        containing the state update of the system state `dx/dt`
        (`dstate[:, :-1]`) and the update of the log density (`dstate[:, -1]`).
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics
        self._n_evals = 0
        
    def forward(self, t, state):
        #print(state)
        *xs, _ = state
        *dxs, div = self._dynamics(t, *xs)
        # return torch.cat([*dxs, -div], dim=-1)

        return (*dxs, -div)


