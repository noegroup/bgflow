import torch
import numpy as np

from .energy import Energy
from .sampling import Sampler
from torch import distributions


__all__ = ["NormalDistribution", "TruncatedNormalDistribution"]


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


class TruncatedNormalDistribution(Energy, Sampler):
    """
    Truncated normal distribution (normal distribution restricted to the interval [lower_bound, upper_bound]
    of dim many independent variables. Used to model molecular angles and bonds.

    Parameters:
        dim : int
            Dimension of the distribution.
        mu : float or tensor of floats of shape (dim, )
            Mean of the untruncated normal distribution.
        sigma : float or tensor of floats of shape (dim, )
            Standard deviation of the untruncated normal distribution.
        lower_bound : float, -np.infty, or tensor of floats of shape (dim, )
            Lower truncation bound.
        upper_bound : float, np.infty, or tensor of floats of shape (dim, )
            Upper truncation bound.
    """
    def __init__(self, dim, mu=0.0, sigma=1.0, lower_bound=0.0, upper_bound=np.infty):
        super(TruncatedNormalDistribution, self).__init__(dim=dim)
        for t in [mu, sigma, lower_bound, upper_bound]:
            assert type(t) is float or type(t) is torch.Tensor
            if type(t) is torch.Tensor:
                assert t.shape in ((1,), (dim,))
        self._dim = dim
        self._mu = mu
        self._sigma = sigma
        self._standard_normal = distributions.normal.Normal(0.0, 1.0)
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._phi_upper = self._standard_normal.cdf((upper_bound-mu)/sigma)
        self._phi_lower = self._standard_normal.cdf((lower_bound-mu)/sigma)

    def _sample(self, n_samples):
        """This is a naive implementation; resample when x falls out of bounds"""
        return self._sample_with_temperature(n_samples, temperature=1)

    def _sample_with_temperature(self, n_samples, temperature):
        sigma = self._sigma * np.sqrt(temperature)
        samples = self._standard_normal.sample((n_samples, self.dim)) * sigma + self._mu
        while True:
            out_of_bounds = (samples > self._upper_bound) | (samples < self._lower_bound)
            if torch.any(out_of_bounds):
                new_samples = self._standard_normal.sample((n_samples, self.dim)) * sigma + self._mu
                samples[out_of_bounds] = new_samples[out_of_bounds]
            else:
                break
        return samples

    def _energy(self, x):
        """The energy is the same as for a untruncated normal distribution (only the partition function changes)."""
        energies = ((x - self._mu) / self._sigma) ** 2  # the sqrt(2) amounts to the 0.5 factor (see return statement)
        energies[x < self._lower_bound] = np.infty
        energies[x > self._upper_bound] = np.infty
        return 0.5 * energies.sum(dim=-1, keepdim=True)

    @property
    def dim(self):
        return self._dim

    def __len__(self):
        return self._dim

