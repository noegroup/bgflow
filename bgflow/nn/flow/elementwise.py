"""Nonlinear One-dimensional Diffeomorphisms"""

import torch
from bgflow.nn.flow.base import Flow


__all__ = ["BentIdentity"]


class BentIdentity(Flow):
    """Bent identity. A nonlinear diffeomorphism with analytic gradients and inverse.
    See https://towardsdatascience.com/secret-sauce-behind-the-beauty-of-deep-learning-beginners-guide-to-activation-functions-a8e23a57d046 .
    """
    def __init__(self):
        super(BentIdentity, self).__init__()

    def _forward(self, x, **kwargs):
        """Forward transform

        Parameters
        ----------
        x : torch.tensor
            Input tensor

        kwargs : dict
            Miscellaneous arguments to satisfy the interface.

        Returns
        -------
        y : torch.tensor
            Elementwise transformed tensor with the same shape as x.

        dlogp : torch.tensor
            Natural log of the Jacobian determinant.
        """
        dlogp = torch.log(self.derivative(x)).sum(dim=-1)[:, None]
        return (torch.sqrt(x ** 2 + 1) - 1) / 2 + x, dlogp

    def _inverse(self, x, **kwargs):
        """Inverse transform

        Parameters
        ----------
        x : torch.tensor
            Input tensor

        kwargs : dict
            Miscellaneous arguments to satisfy the interface.

        Returns
        -------
        y : torch.tensor
            Elementwise transformed tensor with the same shape as x.

        dlogp : torch.tensor
            Natural log of the Jacobian determinant.
        """
        dlogp = torch.log(self.inverse_derivative(x)).sum(dim=-1)[:, None]
        return 2 / 3 * (2 * x + 1 - torch.sqrt(x ** 2 + x + 1)), dlogp

    @staticmethod
    def derivative(x):
        """Elementwise derivative of the activation function."""
        return x / (2 * torch.sqrt(x ** 2 + 1)) + 1

    @staticmethod
    def inverse_derivative(x):
        """Elementwise derivative of the inverse activation function."""
        return 4 / 3 - (2 * x + 1) / (3 * torch.sqrt(x ** 2 + x + 1))
