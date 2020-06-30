"""
Property-preserving parameter perturbations
"""

import torch
from bgtorch.nn.flow.base import Flow
from collections import defaultdict
from collections.abc import Iterable
import warnings

__all__ = ["InvertiblePPPP", "PPPPScheduler"]


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

    def _compute_products(self):
        u = self.u
        v = self.v
        Ainv = self.Ainv
        Ainvu = torch.mv(Ainv, u)
        vtAinvu = torch.dot(v, Ainvu)
        det_update = 1 + vtAinvu
        return Ainvu, vtAinvu, det_update

    @staticmethod
    def _mv_rank_one(u, v, x):
        return torch.einsum("i,j,...j->...i", u, v, x)

    @staticmethod
    def _inv_rank_one(v, Ainv, Ainvu, det_update):
        vtAinv = torch.einsum("i,ij->j", v, Ainv)
        return - 1/det_update * torch.einsum("i,j", Ainvu, vtAinv)

    @staticmethod
    def _inv_mv_rank_one(Ainvy, v, Ainvu, det_update):
        return - 1/det_update * torch.einsum("i,k,...k->...i", Ainvu, v, Ainvy)

    def pppp_merge(self, force_merge=True):
        """PPPP update to hidden parameters.

        Parameters
        ----------
        force_merge : bool
            Whether to update even if the update might hurt numerical stability.

        Returns
        -------
        merged : bool
            Whether a merge was performed.

        """
        with torch.no_grad():
            # never merge nans or infs; instead reset
            if not torch.isfinite(torch.cat([self.u, self.v])).all():
                self.v[:] = torch.randn(self.dim)
                self.u[:] = 0
                return False
            Ainvu, vtAinvu, det_update = (
                self._compute_products()
            )

            # sanity check
            logabsdet_update = torch.log(torch.abs(det_update))
            logabsdet_new = torch.log(torch.abs(det_update*self.detA))
            sane_update = True
            sane_update = sane_update and logabsdet_update > self.min_logdet - 4
            sane_update = sane_update and logabsdet_new > self.min_logdet - 0.5
            sane_update = sane_update and logabsdet_new < self.max_logdet + 0.5
            if sane_update or force_merge:
                self.detA *= det_update
                self.A[:] = self.A + torch.einsum("i,j->ij", self.u, self.v)
                self.Ainv[:] = self.Ainv + self._inv_rank_one(self.v, self.Ainv, Ainvu, det_update)
                self.v[:] = torch.randn(self.dim)
                self.u[:] = 0
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
            Ainvu, vtAinvu, det_update = self._compute_products()
            new_detA = self.detA * det_update
            dlogp = torch.ones_like(x[...,0,None]) * torch.log(torch.abs(new_detA))
            y = torch.einsum("ij,...j->...i", self.A, x) + self._mv_rank_one(self.u, self.v, x)
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
            Ainvu, vtAinvu, det_update = self._compute_products()
            new_detA = self.detA * det_update
            dlogp = - torch.ones_like(y[..., 0, None]) * torch.log(torch.abs(new_detA))
            Ainvy = torch.einsum("ij,...j->...i", self.Ainv, y - self.b)
            x = Ainvy + self._inv_mv_rank_one(Ainvy, self.v, Ainvu, det_update)
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

    def correct(self, recompute_det=False):
        with torch.no_grad():
            self.Ainv[:] = _iterative_solve(self.A, self.Ainv)
            if recompute_det:
                self.detA = torch.det(self.A)*torch.ones_like(self.detA)

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
    """A scheduler for PPPP merges and correction steps.

    Parameters
    ----------
    model : InvertiblePPPP or torch.nn.Module
        A neural net that contains at least one InvertiblePPPP layer.
    optimizer : torch.optim.Optimizer
        An optimizer
    n_force_merge : int
        Number of step() invocations between force merges (PPPP merges even if updates are not sane); None means never
    n_correct : int
        Number of  step() invocations between correction steps; None means never
    n_correct_steps : int
        Number of iterations of the iterative matrix inversion solver
    n_recompute_det : int
        Number of step() invocations between recomputations of the determinant; None means never
    reset_optimizer : bool
        Whether to reset the optimizer after merge.
    """
    def __init__(self, model, optimizer, n_force_merge=10, n_correct=50,
                 n_correct_steps=1, n_recompute_det=None, reset_optimizer=True):
        self._blocks = self._find_invertible_pppp_blocks(model)
        self.optimizer = optimizer
        self.n_force_merge = n_force_merge
        self.n_correct = n_correct
        self.n_correct_steps = n_correct_steps
        self.n_recompute_det = n_recompute_det
        self.reset_optimizer = reset_optimizer
        self.i = 0

    def step(self):
        """Perform a merging step.

        Every `self.n_force_merge` invocations, force merge even if update is not sane.
        Every `self.n_correct` invocations, perform `self.n_correct_steps` many iterative
        inversion steps to improve the inverse matrix.
        Every `self.n_recompute_det` invocations, compute the determinant of the weight matrices from scratch.
        """
        self.i += 1
        merged = []
        for block in self._blocks:
            res = block.pppp_merge(force_merge=self.n_force_merge is not None and self.i % self.n_force_merge == 0)
            merged.append(res)
        if any(merged) and self.reset_optimizer:
            self.optimizer.state = defaultdict(dict)
        if self.n_correct is not None and self.i % self.n_correct == 0:
            for _ in range(self.n_correct_steps):
                for block in self._blocks:
                    block.correct(self.n_recompute_det is not None and self.i % self.n_recompute_det == 0)

    @staticmethod
    def _find_invertible_pppp_blocks(model):
        if isinstance(model, InvertiblePPPP):
            return [model]
        elif isinstance(model, Iterable):
            pppp_blocks = [block for block in model if isinstance(block, InvertiblePPPP)]
            if len(pppp_blocks) == 0:
                warnings.warn("PPPPScheduler not effective. No InvertiblePPPP blocks found in model.")
            return pppp_blocks

    def penalty(self):
        """Sum of penalty functions for all InvertiblePPPP blocks."""
        penalties = [block.penalty() for block in self._blocks]
        return torch.sum(torch.stack(penalties))


_iterative_solve_coefficients = {
    2: (-1., -2.),
    3: (1., 3., -3.),
    7: (1./16., 120., -393., 735., -861., 651., -315., 93., -15.)
}


def _iterative_solve(matrix, inverse_guess, order=7):
    """Perform one iteration of iterative inversion.
    See Soleymani, https://doi.org/10.1155/2012/134653"""
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
