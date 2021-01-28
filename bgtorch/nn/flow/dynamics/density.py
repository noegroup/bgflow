import torch


class DensityDynamics(torch.nn.Module):

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics
        self._n_evals = 0

    def forward(self, t, state):
        """
        Computes the change of the system `dx/dt` at state `x` and
        time `t`. Furthermore, computes the change of density, happening
        due to moving `x` infinitesimally in the direction `dx/dt`
        according to the "instantaneous change of variables rule" [1]
            `dlogp(p(x(t))/dt = -div(dx(t)/dt)`
        [1] Neural Ordinary Differential Equations, Chen et. al,
            https://arxiv.org/abs/1806.07366

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        state : PyTorch tensor
            The current state of the system.
            Consisting of the configuration `xs` and the log density change `dlogp`.

        Returns
        -------
        (*dxs, -dlogp) : Tuple of PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`dxs`) and the update of the log density (`dlogp`).
        """
        xs = state[:-1]
        *dxs, dlogp = self._dynamics(t, *xs)
        return (*dxs, -dlogp)
