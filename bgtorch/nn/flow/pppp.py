"""
Property-preserving parameter perturbations
# TODO: maybe speed this up by avoiding torch.einsum and using torch.func.linear instead
"""

import torch
from bgtorch.nn.flow.base import Flow
from collections import defaultdict

__all__ = ["OrthogonalPPPP", "InvertiblePPPP", "PPPPScheduler"]


class OrthogonalPPPP(Flow):
    """Orthogonal PPPP layer Q*x+b with Givens perturbations

    Notes
    -----
    This does not really work well.

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
        #self.v = torch.nn.Parameter(torch.rand(dim))
        #self.dv = torch.nn.Parameter(torch.zeros(dim))
        self.angles = torch.nn.Parameter(torch.zeros(dim//2))
        self.permutation = torch.randperm(dim)
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

    @staticmethod
    def _nhalfgivens_trafo(x, permutation, angles):
        """Apply n/2 many Givens transformations in parallel"""
        xperm = x[...,permutation]
        s = torch.sin(angles)
        c = torch.cos(angles)
        nhalf = x.shape[-1] // 2
        assert nhalf == angles.shape[0]
        firsthalf = torch.arange(nhalf)
        secondhalf = torch.arange(nhalf, 2*nhalf)
        first = c*xperm[..., firsthalf] + s*xperm[..., secondhalf]
        second = c*xperm[..., secondhalf] - s*xperm[..., firsthalf]
        xperm[..., firsthalf] = first
        xperm[..., secondhalf] = second
        return xperm[..., permutation]

    def pppp_merge(self, force_merge=True):
        """PPPP update to hidden parameters."""
        with torch.no_grad():
            #for v in [self.v, self.v + self.dv]:
            #    self.Q[:] = self.Q - 2*torch.einsum("ij,j,k->ik",self.Q, v, v) / torch.einsum("i,i->", v, v)
            self.Q[:] = self._nhalfgivens_trafo(self.Q, self.permutation, -self.angles)
            #self.v[:] = torch.randn(self.dim)
            #self.dv[:] = 0
            #import numpy as np
            self.angles[:] = 0.0 #torch.rand(self.dim//2)
            self.permutation = torch.randperm(self.dim)

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
        #transformed = self._householder_trafo(self._householder_trafo(x, self.v + self.dv), self.v)
        transformed = self._nhalfgivens_trafo(x, self.permutation, self.angles)
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
        #transformed = self._householder_trafo(self._householder_trafo(x, self.v), self.v + self.dv)
        transformed = self._nhalfgivens_trafo(x, self.permutation, -self.angles)
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
    min_logdet : float
        The minimum for log |det W|.
    max_logdet : float
        The maximum for log |det W|.
    init : str
        Initialization. One of the following
        - "eye": identity matrix,
        - "reverse": reverse order
    """
    def __init__(self, dim, shift=True, penalty_parameter=0.1, min_logdet=-2, max_logdet=15, init="eye"):
        super(InvertiblePPPP, self).__init__()
        self.dim = dim
        self.u = torch.nn.Parameter(torch.zeros(dim))
        self.v = torch.nn.Parameter(torch.randn(dim))
        initial_weight_matrix, initial_inverse, initial_det = {
            "eye": (
                torch.eye(dim),
                torch.eye(dim),
                1.0
            ),
            "reverse": (
                torch.eye(dim)[torch.arange(dim-1,-1,-1)],
                torch.eye(dim)[torch.arange(dim-1,-1,-1)],
                1.0 if dim % 4 < 2 else -1.0
            ),
        }[init]
        self.register_buffer("A", initial_weight_matrix)
        self.register_buffer("Ainv", initial_inverse)
        self.register_buffer("detA", torch.tensor(initial_det, dtype=self.A.dtype))
        if shift:
            self.b = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.register_buffer("min_logdet", min_logdet*torch.ones_like(self.detA))
        self.register_buffer("max_logdet", max_logdet*torch.ones_like(self.detA))
        self.register_buffer("penalty_buffer", torch.zeros_like(self.detA))
        self.register_buffer("penalty_parameter", penalty_parameter*torch.ones_like(self.detA))

    def _compute_scale(self, vtAinvu, detA):
        """Returns one. I was thinking of """
        return torch.ones_like(detA)

    def _compute_products(self):
        u = self.u
        v = self.v
        Ainv = self.Ainv
        detA = self.detA
        Ainvu = torch.mv(Ainv, u)
        vtAinvu = torch.dot(v, Ainvu)
        scale = self._compute_scale(vtAinvu, detA)
        scaled_Ainvu = scale * Ainvu
        scaled_vtAinvu = scale ** 2 * vtAinvu
        det_update = 1 + scaled_vtAinvu
        return scaled_Ainvu, scaled_vtAinvu, scale, det_update

    #def _update_buffers(self):
    #    self.scale = self._determine_scale(vtAinvu, self.detA)
    #    self.det_update = 1.0 + self.scale**2 * vtAinvu
    #    self.Ainvu = Ainvu * self.scale

    @staticmethod
    def _mv_rank_one(u, v, x, scale):
        return scale**2 * torch.einsum("i,j,...j->...i", u, v, x)

    @staticmethod
    def _inv_rank_one(v, scale, Ainv, Ainvu, det_update):
        vtAinv = scale * torch.einsum("i,ij->j", v, Ainv)
        return - 1/det_update * torch.einsum("i,j", Ainvu, vtAinv)

    @staticmethod
    def _inv_mv_rank_one(Ainvy, v, scale, Ainvu, det_update):
        return - 1/det_update * scale * torch.einsum("i,k,...k->...i", Ainvu, v, Ainvy)

    def pppp_merge(self, force_merge=True):
        """PPPP update to hidden parameters."""
        with torch.no_grad():
            # never merge nans or infs; instead reset
            if not torch.isfinite(torch.cat([self.u, self.v])).all():
                self.v[:] = torch.randn(self.dim)
                self.u[:] = 0
                return False
            scaled_Ainvu, scaled_vtAinvu, scale, det_update = (
                self._compute_products()
            )

            # sanity check
            logabsdet_update = torch.log(torch.abs(det_update))
            logabsdet_new = torch.log(torch.abs(det_update*self.detA))
            sane_update = True
            sane_update = sane_update and logabsdet_update > self.min_logdet - 4
            sane_update = sane_update and logabsdet_new > self.min_logdet - 0.5
            sane_update = sane_update and logabsdet_new < self.max_logdet + 0.5
            #print("{:10.4} {:10.4} {}".format(det_update.item(), self.detA.item(), sane_update))
            if not sane_update:
                print("NOT SANE", logabsdet_update.item(), logabsdet_new.item())
            if sane_update or force_merge:
                self.detA *= det_update
                self.A[:] = self.A + scale**2 * torch.einsum("i,j->ij", self.u, self.v)
                self.Ainv[:] = self.Ainv + self._inv_rank_one(self.v, scale, self.Ainv, scaled_Ainvu, det_update)
                self.v[:] = torch.randn(self.dim)
                self.u[:] = 0
                #if det_update < 0:
                #if not sane_update:
                #    self.correct()
                return True
            else:
                return False

    def _buffer_penalty(self, a, b):
        if torch.isclose(self.penalty_parameter, torch.zeros_like(self.penalty_parameter)):
            return self.penalty_parameter
        else:
            self.penalty_buffer = self.penalty_parameter * (
                self._penalty(
                    torch.log(torch.abs(a)),
                    sigma_left=self.min_logdet,
                    sigma_right=self.max_logdet)
                +
                self._penalty(
                    torch.log(torch.abs(b)),
                    sigma_left=self.min_logdet,
                    sigma_right=self.max_logdet)
            )

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
        if self.training:
            scaled_Ainvu, scaled_vtAinvu, scale, det_update = self._compute_products()
            new_detA = self.detA * det_update
            dlogp = torch.ones_like(x[...,0,None]) * torch.log(torch.abs(new_detA))
            y = torch.einsum("ij,...j->...i", self.A, x) + self._mv_rank_one(self.u, self.v, x, scale)
            self._buffer_penalty(det_update, new_detA)
        else:
            dlogp = torch.ones_like(x[..., 0, None]) * torch.log(torch.abs(self.detA))
            y = torch.einsum("ij,...j->...i", self.A, x)
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
        if self.training:
            scaled_Ainvu, scaled_vtAinvu, scale, det_update = self._compute_products()
            new_detA = self.detA * det_update
            dlogp = - torch.ones_like(y[..., 0, None]) * torch.log(torch.abs(new_detA))
            Ainvy = torch.einsum("ij,...j->...i", self.Ainv, y - self.b)
            x = Ainvy + self._inv_mv_rank_one(Ainvy, self.v, scale, scaled_Ainvu, det_update)
            self._buffer_penalty(det_update, new_detA)
        else:
            dlogp = - torch.ones_like(y[..., 0, None]) * torch.log(torch.abs(self.detA))
            x = torch.einsum("ij,...j->...i", self.Ainv, y - self.b)
        return x, dlogp

    def penalty(self):
        """Penalty function to prevent infinite denominator

        p(v, dv) = -penalty_parameter*(1/(norm(1+ v^T A^-1 u)))

        Returns
        -------
        penalty : float
            Value of the penalty function
        """
        if torch.isnan(self.penalty_buffer):
            return torch.ones_like(self.penalty_buffer) * 1e8
        else:
            return self.penalty_buffer

    def correct(self):
        before = torch.mm(self.A, self.Ainv)
        error_before = torch.norm(before - torch.eye(self.dim)).item()
        with torch.no_grad():
            self.Ainv[:] = _iterative_solve(self.A, self.Ainv)
                #self.Ainv[:] = torch.inverse(self.A)
            after = torch.norm(torch.mm(self.A, self.Ainv) - torch.eye(self.dim))
            #if torch.norm(before - torch.eye(self.dim)).item() > 1e-3:
            print(f"{error_before:10.7f} -> {after.item():10.7f}")
            #olddet = torch.det(self.A)
            #self.detA = torch.det(self.A)
            #print(f"{olddet.item():10.7f} -> {self.detA.item():10.7f}")

    @staticmethod
    def _penalty(x, sigma_left=None, sigma_right=None):
        result = torch.zeros_like(x)
        if sigma_left is not None:
            xprime = torch.relu(sigma_left - x)
            result += xprime ** 2
        if sigma_right is not None:
            assert sigma_right > 0
            xprime = torch.relu(x - sigma_right)
            result += xprime ** 2
        return result

    def train(self, mode):
        if torch.norm(self.u) + torch.norm(self.v) > 1e-10:
            self.pppp_merge(force_merge=True)
        return super().train(mode)


class PPPPScheduler:
    def __init__(self, model, optimizer, n_merge=10, n_force_merge=100, n_correct=500, n_correct_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.n_merge = n_merge
        self.n_force_merge = n_force_merge
        self.n_correct = n_correct
        self.n_correct_steps = n_correct_steps
        self.i = 0

    def step(self):
        self.i += 1
        if self.i % self.n_merge == 0:
            self.model.pppp_merge(force_merge=self.i % self.n_force_merge == 0)
            self.optimizer.state = defaultdict(dict)
        if self.i % self.n_correct == 0:
            for _ in range(self.n_correct_steps):
                self.model.correct()


_iterative_solve_coefficients = {
    2: (-1., -2.),
    3: (1., 3., -3.),
    7: (1./16., 120., -393., 735., -861., 651., -315., 93., -15.)
}


def _iterative_solve(matrix, inverse_guess, order=7):
    """see Soleymani, https://doi.org/10.1155/2012/134653"""
    coeffs = _iterative_solve_coefficients[order]
    factor = coeffs[:2]
    coeffs = coeffs[2:]
    error = torch.mm(matrix, inverse_guess)
    correction = error.clone()
    indices = torch.arange(matrix.shape[0])
    for c in reversed(coeffs):
        correction[indices, indices] += c
        correction = torch.mm(error, correction)
    correction[indices, indices] += factor[1]
    return factor[0] * torch.mm(inverse_guess, correction)
