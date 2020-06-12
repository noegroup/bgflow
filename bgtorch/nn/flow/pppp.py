"""
Property-preserving parameter perturbations
# TODO: maybe speed this up by avoiding torch.einsum and using torch.func.linear instead
"""

import torch
from bgtorch.nn.flow.base import Flow

__all__ = ["OrthogonalPPPP", "InvertiblePPPP"]


class OrthogonalPPPP(Flow):
    """Orthogonal PPPP layer Q*x+b with double-Householder perturbations

    Attributes
    ----------
    dim : int
        dimension
    shift : boolean
        Whether to use a shift parameter (+b). If False, b=0.
    penalty_parameter : float
        Scaling factor for the well-definedness of the double-Householder update.
    """
    def __init__(self, dim, shift=True, penalty_parameter=0.0):
        super(OrthogonalPPPP, self).__init__()
        self.dim = dim
        self.v = torch.nn.Parameter(torch.rand(dim))
        self.dv = torch.nn.Parameter(torch.zeros(dim))
        self.register_buffer("Q", torch.eye(dim))
        if shift:
            self.b = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.register_buffer("penalty_parameter", torch.tensor(penalty_parameter))

    @staticmethod
    def _householder_trafo(x, v):
        dx = - 2 * torch.einsum("j,i,...i->...j", v, v, x) / torch.einsum("i,i->", v, v)
        return x + dx

    def pppp_merge(self):
        """PPPP update to hidden parameters."""
        with torch.no_grad():
            for v in [self.v, self.v + self.dv]:
                self.Q[:] = self.Q - 2*torch.einsum("ij,j,k->ik",self.Q, v, v) / torch.einsum("i,i->", v, v)
            self.v[:] = torch.randn(self.dim)
            self.dv[:] = 0

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
        dlogp = torch.zeros_like(x[...,0,None])
        transformed = self._householder_trafo(self._householder_trafo(x, self.v + self.dv), self.v)
        y = torch.einsum("ab,...b->...a", self.Q, transformed)
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
        dlogp = torch.zeros_like(y[...,0,None])
        x = torch.einsum("ba,...b->...a", self.Q, y - self.b)
        transformed = self._householder_trafo(self._householder_trafo(x, self.v), self.v + self.dv)
        return transformed, dlogp

    def penalty(self):
        """Penalty function to prevent infinite denominator

        p(v, dv) = penalty_parameter*(1/(||v||) + 1/(||v-dv||))

        Returns
        -------
        penalty : float
            Value of the penalty function
        """
        #constraint = 1/(torch.norm(self.v)) + 1/(torch.norm(self.v-self.dv))  # != 0
        return self.penalty_parameter * 0.0
        #return self.penalty_parameter * torch.norm(constraint)


class InvertiblePPPP(Flow):
    """Invertible PPPP layer A*x+b with rank-one perturbations

    Attributes
    ----------
    dim : int
        dimension
    shift : boolean
        Whether to use a shift parameter (+b). If False, b=0.
    penalty_parameter : float
        Scaling factor for the regularity constraint.
    """
    def __init__(self, dim, shift=True, penalty_parameter=0.1, min_logdet=-2.0, max_logdet=5.0):
        super(InvertiblePPPP, self).__init__()
        self.dim = dim
        self.u = torch.nn.Parameter(torch.zeros(dim))
        self.v = torch.nn.Parameter(torch.randn(dim))
        self.register_buffer("A", torch.eye(dim))
        self.register_buffer("Ainv", torch.eye(dim))
        self.register_buffer("logdetA", torch.tensor(0.0))
        if shift:
            self.b = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.register_buffer("penalty_parameter", torch.tensor(penalty_parameter))
        self.register_buffer("min_logdet", torch.tensor(min_logdet))
        self.register_buffer("max_logdet", torch.tensor(max_logdet))

    @staticmethod
    def _logdet_rank_one(u, v, Ainv):
        return torch.log(1.0+torch.einsum("i,ij,j->", v, Ainv, u))

    @staticmethod
    def _mv_rank_one(u, v, x):
        return torch.einsum("i,j,...j->...i", u, v, x)

    @staticmethod
    def _inv_rank_one(u, v, Ainv):
        vtAinv = torch.einsum("i,ij->j", v, Ainv)
        denominator = 1 + torch.einsum("i,i->", vtAinv, u)
        return - 1/denominator * torch.einsum("ij,j,k", Ainv, u, vtAinv)

    @staticmethod
    def _inv_mv_rank_one(u, v, Ainv, Ainvy):
        Ainvu = torch.einsum("ij,j->i", Ainv, u)
        denominator = 1 + torch.einsum("i,i->", v, Ainvu)
        return - 1/denominator * torch.einsum("i,k,...k->...i", Ainvu, v, Ainvy)

    def pppp_merge(self):
        """PPPP update to hidden parameters."""
        with torch.no_grad():
            self.logdetA += self._logdet_rank_one(self.u, self.v, self.Ainv)
            self.A[:] = self.A + torch.einsum("i,j->ij", self.u, self.v)
            self.Ainv[:] = self.Ainv + self._inv_rank_one(self.u, self.v, self.Ainv)
            self.u[:] = 0
            self.v[:] = torch.randn(self.dim)

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
        new_logdet = self.logdetA + self._logdet_rank_one(self.u, self.v, self.Ainv)
        #print(new_logdet)
        dlogp = torch.ones_like(x[...,0,None]) * new_logdet
        y = torch.einsum("ij,...j->...i", self.A, x) + self._mv_rank_one(self.u, self.v, x)
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
        dlogp = - torch.ones_like(y[..., 0, None]) * (self.logdetA + self._logdet_rank_one(self.u, self.v, self.Ainv))
        Ainvy = torch.einsum("ij,...j->...i", self.Ainv, y - self.b)
        x = Ainvy + self._inv_mv_rank_one(self.u, self.v, self.Ainv, Ainvy)
        return x, dlogp

    def penalty(self):
        """Penalty function to prevent infinite denominator

        p(v, dv) = -penalty_parameter*(1/(norm(1+ v^T A^-1 u)))

        Returns
        -------
        penalty : float
            Value of the penalty function
        """
        # unnecessary to compute this twice; make a buffer
        new_logdet = self.logdetA + self._logdet_rank_one(self.u, self.v, self.Ainv)
        if new_logdet > self.max_logdet:
            return self.penalty_parameter*(new_logdet - self.max_logdet)**2
        elif new_logdet < self.min_logdet:
            return self.penalty_parameter*(self.min_logdet - new_logdet)**2
        else:
            return torch.zeros_like(self.penalty_parameter)
        #else:
        #    return self.penalty_parameter / torch.exp(new_logdet) - self.penalty_parameter / self.min_determinant
