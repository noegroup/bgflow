"""
(Pseudo-) Orthogonal Linear Layers. Advantage: Jacobian determinant is unity.
"""

import torch
from bgflow.nn.flow.base import Flow

__all__ = ["PseudoOrthogonalFlow"]

# Note: OrthogonalPPPP is implemented in pppp.py


class PseudoOrthogonalFlow(Flow):
    """Linear flow W*x+b with a penalty function
        penalty_parameter*||W^T W - I||^2

    Attributes
    ----------
    dim : int
        dimension
    shift : boolean
        Whether to use a shift parameter (+b). If False, b=0.
    penalty_parameter : float
        Scaling factor for the orthogonality constraint.
    """
    def __init__(self, dim, shift=True, penalty_parameter=1e5):
        super(PseudoOrthogonalFlow, self).__init__()
        self.dim = dim
        self.W = torch.nn.Parameter(torch.eye(dim))
        if shift:
            self.b = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.register_buffer("penalty_parameter", torch.tensor(penalty_parameter))

    def _forward(self, x, **kwargs):
        """Forward transform.

        Attributes
        ----------
        x : torch.tensor
            The input vector. The transform is applied to the last dimension.
        kwargs : dict
            keyword arguments to satisfy the interface

        Returns
        -------
        y : torch.tensor
            W*x + b
        dlogp : torch.tensor
            natural log of the Jacobian determinant
        """
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        y = torch.einsum("ab,...b->...a", self.W, x)
        return y + self.b, dlogp

    def _inverse(self, y, **kwargs):
        """Inverse transform assuming that W is orthogonal.

        Attributes
        ----------
        y : torch.tensor
            The input vector. The transform is applied to the last dimension.
        kwargs : dict
            keyword arguments to satisfy the interface

        Returns
        -------
         x : torch.tensor
            W^T*(y-b)
        dlogp : torch.tensor
            natural log of the Jacobian determinant
        """
        dlogp = torch.zeros(*y.shape[:-1], 1).to(y)
        x = torch.einsum("ab,...b->...a", self.W.transpose(1, 0), y - self.b)
        return x, dlogp

    def penalty(self):
        """Penalty function for the orthogonality constraint

        p(W) = penalty_parameter * ||W^T*W - I||^2.

        Returns
        -------
        penalty : float
            Value of the penalty function
        """
        return self.penalty_parameter * torch.sum((torch.eye(self.dim) - torch.mm(self.W.transpose(1, 0), self.W)) ** 2)
