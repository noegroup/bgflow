import torch


class BlackBoxDynamics(torch.nn.Module):
    """Black box dynamics that allows to use any dynamics function.
    The divergence of the dynamics is computed with a divergence estimator.
    """

    def __init__(self, dynamics_function, divergence_estimator, compute_divergence=True):
        super().__init__()
        self._dynamics_function = dynamics_function
        self._divergence_estimator = divergence_estimator
        self._compute_divergence = compute_divergence

    def forward(self, t, *xs):
        """
        Computes the change of the system `dxs` at state `xs` and
        time `t`. Furthermore, can also compute the change of log density
        which is equal to the divergence of the change.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        xs : PyTorch tensor
            The current configuration of the system

        Returns
        -------
        (*dxs, divergence): Tuple of PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`dxs`) and the update of the log density (`dlogp`)
        """
        if self._compute_divergence:
            *dxs, divergence = self._divergence_estimator(
                self._dynamics_function, t, *xs
            )
        else:
            dxs = self._dynamics_function(t, xs)
            divergence = None
        return (*dxs, divergence)
