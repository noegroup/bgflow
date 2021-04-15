import torch


class TimeIndependentDynamics(torch.nn.Module):
    """
    Time independent dynamics function.
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics

    def forward(self, t, xs):
        """
        Computes the change of the system `dxs` due to a time independent dynamics function.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        xs : PyTorch tensor
            The current configuration of the system

        Returns
        -------
        dxs : PyTorch tensor
            The change of the system due to the dynamics function
        """

        dxs = self._dynamics(xs)
        return dxs
