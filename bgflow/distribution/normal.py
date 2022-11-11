import torch
import numpy as np
from torch.distributions import VonMises, Independent
import warnings

from .energy.base import Energy
from .sampling.base import Sampler


__all__ = ["NormalDistribution", "TruncatedNormalDistribution", "MeanFreeNormalDistribution", "CircularNormalDistribution"]


def _is_symmetric_matrix(m):
    return torch.allclose(m, m.t())


class NormalDistribution(Energy, Sampler):
    def __init__(self, dim, mean=None, cov=None):
        super().__init__(dim=dim)
        self._has_mean = mean is not None
        if self._has_mean:
            assert len(mean.shape) == 1, "`mean` must be a vector"
            assert mean.shape[-1] == self.dim, "`mean` must have dimension `dim`"
            self.register_buffer("_mean", mean)
        else:
            self.register_buffer("_mean", torch.zeros(self.dim))
        self._has_cov = False
        if cov is not None:
            self.set_cov(cov)

    def energy(self, x, temperature=1.0):
        if self._has_mean:
            x = x - self._mean
        if self._has_cov:
            diag = torch.exp(-0.5 * self._log_diag)
            x = x @ self._rot
            x = x * diag
        x = x / (temperature ** 0.5)
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) + self._log_Z(temperature)

    def _log_Z(self, temperature=1.0):
        temperature = torch.as_tensor(temperature, dtype=self._mean.dtype, device=self._mean.device)
        log_z = self.dim / 2 * torch.log(2 * np.pi * temperature)
        if self._has_cov:
            log_z = log_z + 1 / 2 * self._log_diag.sum()
        return log_z

    @staticmethod
    def _eigen(cov):
        try:
            diag, rot = torch.linalg.eig(cov)
            assert (diag.imag.abs() < 1e-6).all(), "`cov` possesses complex valued eigenvalues"
            diag, rot = diag.real, rot.real
        except AttributeError:
            # old implementation
            diag, rot = torch.eig(cov, eigenvectors=True)
            assert (diag[:,1].abs() < 1e-6).all(), "`cov` possesses complex valued eigenvalues"
            diag = diag[:,0] 
        return diag + 1e-6, rot
            
    def set_cov(self, cov):
        self._has_cov = True
        assert (
            len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]
        ), "`cov` must be square matrix"
        assert (
            cov.shape[0] == self.dim and cov.shape[1] == self.dim
        ), "`cov` must have dimension `[dim, dim]`"
        assert _is_symmetric_matrix, "`cov` must be symmetric"
        diag, rot = self._eigen(cov)
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
            samples = samples * (temperature ** 0.5)
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

    Parameters
    ----------
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
        If "icdf", sample by passing a uniform sample through the inverse cdf.
        If "rejection", sample by rejecting normal distributed samples that fall out of bounds.
    is_learnable : bool
        Whether sigma and mu are learnable parameters.
    """
    def __init__(
        self,
        mu,
        sigma=torch.tensor(1.0),
        lower_bound=torch.tensor(0.0),
        upper_bound=torch.tensor(np.infty),
        assert_range=True,
        sampling_method="icdf",
        is_learnable=False
    ):
        for t in [mu, sigma, lower_bound, upper_bound]:
            assert type(t) is torch.Tensor
            if type(t) is torch.Tensor:
                assert t.shape in (torch.Size([]), (1,), mu.shape)

        super().__init__(dim=mu.shape)

        if is_learnable:
            self._mu = torch.nn.Parameter(mu)
            self._logsigma = torch.nn.Parameter(torch.log(sigma.to(mu)))
        else:
            self.register_buffer("_mu", mu)
            self.register_buffer("_logsigma", torch.log(sigma.to(mu)))
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
            self.register_buffer("_cdf_lower_bound", self._standard_normal.cdf(alpha.detach()))
            self.register_buffer("_cdf_upper_bound", self._standard_normal.cdf(beta.detach()))
        else:
            raise ValueError(f'Unknown sampling method "{sampling_method}"')

    @property
    def _sigma(self):
        return torch.exp(self._logsigma)

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

    def icdf(self, x):
        r = self.Z * x + self._cdf_lower_bound
        return self._standard_normal.icdf(r) * self._sigma + self._mu

    def cdf(self, x):
        return (self._standard_normal.cdf((x - self._mu)/self._sigma) - self._cdf_lower_bound)/self.Z

    def log_prob(self, x):
        return self._standard_normal.log_prob((x - self._mu)/self._sigma) - torch.log(self.Z * self._sigma)

    @property
    def Z(self):
        return self._cdf_upper_bound - self._cdf_lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def dim(self):
        return self._dim

    def __len__(self):
        return self._dim


class MeanFreeNormalDistribution(Energy, Sampler):
    """ Mean-free normal distribution. """

    def __init__(self, dim, n_particles, std=1., two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._two_event_dims = two_event_dims
        self._dim = dim
        self._n_particles = n_particles
        self._spacial_dims = dim // n_particles
        self.register_buffer("_std", torch.as_tensor(std))

    def _energy(self, x):
        x = self._remove_mean(x).view(-1, self._dim)
        # TODO: make consistent
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) / self._std ** 2

    def sample(self, n_samples, temperature=1.):
        x = torch.ones((n_samples, self._n_particles, self._spacial_dims), dtype=self._std.dtype,
                         device=self._std.device).normal_(mean=0, std=self._std)
        x = self._remove_mean(x)
        if not self._two_event_dims:
            x = x.view(-1, self._dim)
        return x

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x


class CircularNormalDistribution(Energy, Sampler):
# TODO: test
    """ Wrapper for pytorch VonMises distribution.
    Output is mapped to the [0,1] interval instead of the original [-pi,pi].

    Sampling from this distribution becomes extremely slow for big values of sigma (above 1),
    when the distribution becomes almost uniform over the [0,1] interval."""

    def __init__(self, mu: torch.Tensor, sigma):
        assert type(mu) is torch.Tensor
        if torch.any(torch.as_tensor(sigma > 3.6)):
            warnings.warn("CircularNormalDistribution with sigma greater than 1 can be quite slow."
              "Use UniformDistribution instead")
        loc = 2 * np.pi * (mu - 0.5)
        concentration = (2 * np.pi * sigma)**(-2)
        self._delegate = Independent(VonMises(loc, concentration), 1)
        super().__init__(dim=mu.shape)

    def _sample(self, n_samples):
        if isinstance(n_samples, int):
            event_shape = torch.Size([n_samples])
        else:
            event_shape = torch.Size(n_samples)
        return self._delegate.sample(event_shape) / (2 * np.pi) + 0.5

    def _sample_with_temperature(self, n_samples, temperature):
        # TODO is this trivial?
        raise NotImplementedError()

    def _energy(self, x):
        return -self._delegate.log_prob(2 * np.pi * (x - 0.5))

    def __getattr__(self, name):
        try:
            return getattr(self._delegate, name)
        except AttributeError as e:
            msg = str(e)
            msg = msg.replace(self._delegate.__class__.__name__, "CircularNormalDistribution")
            raise AttributeError(msg)
