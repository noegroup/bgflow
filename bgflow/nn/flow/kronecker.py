import torch
import numpy as np


from .base import Flow


# TODO: write docstrings


def _is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)


def _kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(
        A.size(0) * B.size(0), A.size(1) * B.size(1)
    )


def _batch_determinant_2x2(As, log=False):
    result = As[:, 0, 0] * As[:, 1, 1] - As[:, 1, 0] * As[:, 0, 1]
    if log:
        result = result.abs().log()
    return result


def _create_ortho_matrices(n, d):
    qs = []
    for i in range(n):
        q, _ = np.linalg.qr(np.random.normal(size=(d, d)))
        qs.append(q)
    qs = np.array(qs)
    return qs


class KroneckerProductFlow(Flow):
    def __init__(self, n_dim):
        super().__init__()

        assert _is_power2(n_dim)

        self._n_dim = n_dim
        self._n_factors = int(np.log2(n_dim))

        self._factors = torch.nn.Parameter(
            torch.Tensor(_create_ortho_matrices(self._n_factors, 2))
        )
        self._bias = torch.nn.Parameter(torch.Tensor(1, n_dim).zero_())

    def _forward(self, x, **kwargs):
        n_batch = x.shape[0]
        factors = self._factors.to(x)
        M = factors[0]
        dets = _batch_determinant_2x2(factors)
        det = dets[0]
        power = 2
        for new_det, factor in zip(dets[1:], factors[1:]):
            det = det.pow(2) * new_det.pow(power)
            M = _kronecker(M, factor)
            power = power * 2
        dlogp = torch.zeros(n_batch, 1).to(x)
        dlogp = dlogp + det.abs().log().sum(dim=-1, keepdim=True)
        return x @ M + self._bias.to(x), dlogp

    def _inverse(self, x, **kwargs):
        n_batch = x.shape[0]
        factors = self._factors.to(x)
        inv_factors = torch.inverse(factors)
        M = inv_factors[0]
        inv_dets = _batch_determinant_2x2(inv_factors)
        inv_det = inv_dets[0]
        power = 2
        for new_inv_det, factor in zip(inv_dets[1:], inv_factors[1:]):
            inv_det = inv_det.pow(2) * new_inv_det.pow(power)
            M = _kronecker(M, factor)
            power = power * 2
        dlogp = torch.zeros(n_batch, 1).to(x)
        dlogp = dlogp + inv_det.abs().log().sum(dim=-1, keepdim=True)
        return (x - self._bias.to(x)) @ M, dlogp
