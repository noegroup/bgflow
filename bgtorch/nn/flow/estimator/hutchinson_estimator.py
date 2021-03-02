import torch


class HutchinsonEstimator(torch.nn.Module):
    """
    Estimation of the divergence of a dynamics function with the Hutchinson Estimator [1].
    [1] A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines, Hutchinson
    """

    def __init__(self, rademacher=True):
        super().__init__()
        self._rademacher = rademacher
        self._reset_noise = True

    def reset_noise(self, reset_noise=True):
        """
        Resets the noise vector.
        """

        self._reset_noise = reset_noise

    def forward(self, dynamics, t, xs):
        """
        Computes the change of the system `dxs` due to a time independent dynamics function.
        Furthermore, also estimates the change of log density, which is equal to the divergence of the change `dxs`,
        with the Hutchinson Estimator.
        This is done with either Rademacher or Gaussian noise.

        Parameters
        ----------
        dynamics : torch.nn.Module
            A dynamics function that computes the change of the system and its density.
        t : PyTorch tensor
            The current time
        xs : PyTorch tensor
            The current configuration of the system

        Returns
        -------
        dxs, -divergence: PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`dxs`) and the negative update of the log density (`-divergence`)
        """

        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            dxs = dynamics(t, xs)

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            system_dim = dxs.shape[-1]

            if self._reset_noise == True:
                self._reset_noise = False
                if self._rademacher == True:
                    self._noise = torch.randint(low=0, high=2, size=xs.shape).to(xs) * 2 - 1
                else:
                    self._noise = torch.randn_like(xs)

            noise_ddxs = torch.autograd.grad(dxs, xs, self._noise, create_graph=True)[0]
            divergence = torch.sum((noise_ddxs * self._noise).view(-1, system_dim), 1, keepdim=True)

        return dxs, -divergence
