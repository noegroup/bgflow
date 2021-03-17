import torch
import numpy as np

from .energy.base import Energy
from .sampling.base import Sampler


__all__ = ["NormalDistribution", "TruncatedNormalDistribution"]


def _is_symmetric_matrix(m):
    return torch.allclose(m, m.t())


class NormalDistribution(Energy, Sampler):

    def __init__(self, dim, mean=None, cov=None):
        super().__init__(dim=dim)
        self._has_mean = mean is not None
        if self._has_mean:
            assert len(mean.shape) == 1, "`mean` must be a vector"
            assert mean.shape[-1] == self.dim, "`mean` must have dimension `dim`"
            self.register_buffer("_mean", mean.unsqueeze(0))
        else:
            self.register_buffer("_mean", torch.zeros(self.dim))
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
        self._log_Z = self.dim / 2 * np.log(2 * np.pi)
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
        samples = torch.randn(n_samples, self.dim, dtype=self._mean.dtype, device=self._mean.device)
        if self._has_cov:
            samples = samples.to(self._rot)
            inv_diag = torch.exp(0.5 * self._log_diag)
            samples = samples * inv_diag
            samples = samples @ self._rot.t()
        if isinstance(temperature, torch.Tensor):
            samples = samples * temperature.sqrt()
        else:
            samples = samples * np.sqrt(temperature)
        if self._has_mean:
            samples = samples.to(self._mean)
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
        assert_range : bool
            Whether to raise an error when `energy` is called on input that falls out of bounds.
            Otherwise the energy is set to infinity.
        sampling_method : str
            If "icdf", sample by passing a uniform sample through the inverse cdf (only available when
            the bounds are both finite). When one of the bounds is infinite, the sampling method
            is automatically set to "rejection".
            If "rejection", sample by rejecting normal distributed samples that fall out of bounds.
    """

    def __init__(
        self,
        mu,
        sigma=torch.tensor(1.0),
        lower_bound=torch.tensor(0.0),
        upper_bound=torch.tensor(np.infty),
        assert_range=True,
        sampling_method="icdf",
    ):
        for t in [mu, sigma, lower_bound, upper_bound]:
            assert type(t) is torch.Tensor
            if type(t) is torch.Tensor:
                assert t.shape in (torch.Size([]), (1,), mu.shape)

        super().__init__(dim=mu.shape)

        if not (torch.isfinite(lower_bound).all() and torch.isfinite(upper_bound).all()):
            sampling_method = "rejection"

        self.register_buffer("_mu", mu)
        self.register_buffer("_sigma", sigma.to(mu))
        self.register_buffer("_upper_bound", upper_bound.to(mu))
        self.register_buffer("_lower_bound", lower_bound.to(mu))
        self.assert_range = assert_range

        self._standard_normal = torch.distributions.normal.Normal(torch.tensor(0.0).to(mu), torch.tensor(1.0).to(mu))

        if sampling_method == "rejection":
            self._sample_impl = self._rejection_sampling
        elif sampling_method == "icdf":
            self._sample_impl = self._icdf_sampling
            alpha = (self._lower_bound - self._mu) / self._sigma
            beta = (self._upper_bound - self._mu) / self._sigma
            self.register_buffer("_cdf_lower_bound", self._standard_normal.cdf(alpha))
            self.register_buffer("_cdf_upper_bound", self._standard_normal.cdf(beta))
        else:
            raise ValueError(f'Unknown sampling method "{sampling_method}"')

    def _sample(self, n_samples):
        return self._sample_with_temperature(n_samples, temperature=1)

    def _rejection_sampling(self, n_samples, temperature):
        sigma = self._sigma * np.sqrt(temperature)
        rejected = torch.ones(n_samples, device=self._mu.device, dtype=bool)
        samples = torch.empty(n_samples, *self.event_shape, device=self._mu.device, dtype=self._mu.dtype)
        while True:
            n_rejected = (rejected).long().sum()
            samples[rejected] = torch.randn(
                n_rejected, *self.event_shape, device=self._mu.device, dtype=self._mu.dtype
            ) * sigma + self._mu
            rejected = torch.any(
                ((samples > self._upper_bound) | (samples < self._lower_bound)).view(n_samples, -1),
                dim=-1
            )
            if not torch.any(rejected):
                break
        return samples

    def _icdf_sampling(self, n_samples, temperature):
        sigma = self._sigma * np.sqrt(temperature)
        u = torch.rand(n_samples, *self.event_shape).to(self._mu)
        r = (self._cdf_upper_bound - self._cdf_lower_bound) * u + self._cdf_lower_bound
        x = self._standard_normal.icdf(r) * sigma + self._mu
        return x

    def _sample_with_temperature(self, n_samples, temperature):
        return self._sample_impl(n_samples, temperature)

    def _energy(self, x):
        """The energy is the same as for a untruncated normal distribution
        (only the partition function changes).

        Raises
        ------
        ValueError
            If input is out of bounds and assert_ranges is True.
        """
        energies = (
            (x - self._mu) / self._sigma
        ) ** 2  # the sqrt(2) amounts to the 0.5 factor (see return statement)
        if self.assert_range:
            if (x < self._lower_bound).any() or (x > self._upper_bound).any():
                raise ValueError("input out of bounds")
        else:
            energies[x < self._lower_bound] = np.infty
            energies[x > self._upper_bound] = np.infty
        return 0.5 * energies.sum(dim=-1, keepdim=True)

    @property
    def dim(self):
        return self._dim

    def __len__(self):
        return self._dim
