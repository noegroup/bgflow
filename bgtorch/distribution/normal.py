import torch
import numpy as np

from .energy import Energy
from .sampling import Sampler


def _is_symmetric_matrix(m):
    return torch.allclose(m, m.t())


class NormalDistribution(Energy, Sampler):
    def __init__(self, dim, mean=None, cov=None):
        super().__init__(dim)
        self._has_mean = mean is not None
        if self._has_mean:
            assert len(mean.shape) == 1, "`mean` must be a vector"
            assert mean.shape[-1] == self.dim, "`mean` must have dimension `dim`"
            self.register_buffer("_mean", mean.unsqueeze(0))
        self._has_cov = False
        if cov is not None:
            self.set_cov(cov)
        self._compute_Z()

    def _energy(self, x):
        if self._has_mean:
            x = x - self._mean
        if self._has_cov:
            diag = torch.exp(-0.5 * self._log_diag)
            x = x @ self._rot
            x = x * diag
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) + self._log_Z

    def _compute_Z(self):
        self._log_Z = self._dim / 2 * np.log(2 * np.pi)
        if self._has_cov:
            self._log_Z += 1 / 2 * self._log_diag.sum()  # * torch.slogdet(cov)[1]

    def set_cov(self, cov):
        self._has_cov = True
        assert (
            len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]
        ), "`cov` must be square matrix"
        assert (
            cov.shape[0] == self.dim and cov.shape[1] == self.dim
        ), "`cov` must have dimension `[dim, dim]`"
        assert _is_symmetric_matrix, "`cov` must be symmetric"
        diag, rot = torch.eig(cov, eigenvectors=True)
        assert torch.allclose(
            diag[:, 1], torch.zeros_like(diag[:, 1])
        ), "`cov` possesses complex valued eigenvalues"
        diag = diag[:, 0] + 1e-6
        assert torch.all(diag > 0), "`cov` must be positive definite"
        self.register_buffer("_log_diag", diag.log().unsqueeze(0))
        self.register_buffer("_rot", rot)

    def _sample_with_temperature(self, n_samples, temperature=1.0):
        samples = torch.Tensor(n_samples, self._dim).normal_()
        if self._has_cov:
            samples.to(self._rot)
            inv_diag = torch.exp(0.5 * self._log_diag)
            samples = samples * inv_diag
            samples = samples @ self._rot.t()
        samples = samples * np.sqrt(temperature)
        if self._has_mean:
            samples.to(self._mean)
            samples = samples + self._mean
        return samples

    def _sample(self, n_samples):
        return self._sample_with_temperature(n_samples)
