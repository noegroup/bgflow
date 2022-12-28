

import torch
from bgflow.nn.flow.base import Flow


__all__ = ["TriuFlow"]


class TriuFlow(Flow):
    """Linear flow (I+R)*x+b with a upper triangular matrix R.

    Attributes
    ----------
    dim : int
        dimension
    shift : boolean
        Whether to use a shift parameter (+b). If False, b=0.
    """
    def __init__(self, dim, shift=True):
        super(TriuFlow, self).__init__()
        self.dim = dim
        self.register_buffer("indices", torch.triu_indices(dim, dim))
        n_matrix_parameters = self.indices.shape[1]
        self._unique_elements = torch.nn.Parameter(torch.zeros(n_matrix_parameters))
        if shift:
            self.b = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.register_buffer("R", torch.zeros((self.dim, self.dim)))

    def _make_r(self):
        self.R[:] = 0
        self.R[self.indices[0], self.indices[1]] = self._unique_elements
        self.R += torch.eye(self.dim)
        return self.R

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
        R = self._make_r()
        dlogp = torch.ones_like(x[...,0,None])*torch.sum(torch.log(torch.abs(torch.diagonal(R))))
        y = torch.einsum("ab,...b->...a", R, x)
        return y + self.b, dlogp

    def _inverse(self, y, **kwargs):
        """Inverse transform.

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
        R = self._make_r()
        dlogp = torch.ones_like(y[...,0,None])*(-torch.sum(torch.log(torch.abs(torch.diagonal(R)))))
        try:
            x = torch.linalg.solve_triangular(R, (y-self.b)[...,None], upper=True)
        except AttributeError:
            # legacy call for torch < 1.11
            x, _ = torch.triangular_solve((y-self.b)[...,None], R)
        return x[...,0], dlogp
