
import torch
import numpy as np
from .base import Flow
from .sequential import SequentialFlow
from .inverted import InverseFlow
from ...distribution.normal import TruncatedNormalDistribution


__all__ = ["CDFTransform", "DistributionTransferFlow", "ConstrainGaussianFlow"]


class CDFTransform(Flow):
    """A transform that is defined by the CDF of a distribution.
    The transform maps from the distribution's support to [0,1].

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        The torch distribution (or some other object that implements cdf, icdf, and log_prog).
    eps : float
        Limit to ensure that cdf values are strictly in (0,1). Log deterimants are enforced > -1/eps.
    """
    def __init__(self, distribution, eps=1e-7):
        super().__init__()
        self.distribution = distribution
        self._eps = eps

    def _forward(self, x, *args, **kwargs):
        y = self.distribution.cdf(x)
        if self._eps is not None:
            y = y.clamp(self._eps, 1. - self._eps)
        logdet = self.distribution.log_prob(x)
        if self._eps is not None:
            logdet = logdet.clamp_min(-1/self._eps)

        return y, logdet.sum(dim=-1, keepdim=True)

    def _inverse(self, x, *args, **kwargs):
        if self._eps is not None:
            x = x.clamp(self._eps, 1. - self._eps)
        y = self.distribution.icdf(x)
        logdet = -self.distribution.log_prob(y)
        if self._eps is not None:
            logdet = logdet.clamp_min(-1/self._eps)
        return y, logdet.sum(dim=-1, keepdim=True)


class DistributionTransferFlow(SequentialFlow):
    """Transfer a sample from one distribution to another.

    Parameters
    ----------
    source_distribution : torch.distributions.Distribution
        Distribution of input data.
    target_distribution : torch.distributions.Distribution
        Distribution of output data.
    """
    def __init__(self, source_distribution, target_distribution, eps=1e-7):
        super().__init__([
            CDFTransform(source_distribution, eps=eps),
            InverseFlow(CDFTransform(target_distribution, eps=eps))
        ])


class ConstrainGaussianFlow(Flow):
    """Constrain a variable to a specified range
    through the CDF of a Gaussian and the inverse CDF of a truncated Gaussian.

    Parameters
    ----------
    mu : torch.Tensor
        Parameter of the normal distribution.
    sigma : torch.Tensor
        Parameter of the normal distribution.
    lower_bound : float
        Lower bound for the output
    upper_bound : float
        Upper bound for the output
    assert_range : bool
        Whether to raise an Exception when the input to the truncated normal distribution falls out of bounds
    mu_out : torch.Tensor or None
        If specified, this is the shift of the truncated normal distribution. Otherwise, mu_out = mu.
    sigma_out : torch.Tensor or None
        If specified, this is the scale of the truncated normal distribution. Otherwise, sigma_out = sigma.
    eps : float
        If not None, clamp all CDF values to [0+eps, 1-eps] and clamp log_prob values > -1/eps.
    """
    def __init__(
        self,
        mu,
        sigma=torch.tensor(1.0),
        lower_bound=0.0,
        upper_bound=np.infty,
        assert_range=True,
        mu_out=None,
        sigma_out=None,
        eps=1e-7
    ):
        super().__init__()
        source = torch.distributions.Normal(mu, sigma.to(mu))
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
        target = TruncatedNormalDistribution(
            mu=mu if mu_out is None else mu_out.to(mu),
            sigma=sigma if sigma_out is None else sigma_out.to(mu),
            lower_bound=lower_bound*torch.ones_like(mu),
            upper_bound=upper_bound*torch.ones_like(mu),
            assert_range=assert_range
        )
        self._trafo = DistributionTransferFlow(source, target, eps)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def _forward(self, x, *args, **kwargs):
        y, dlogp = self._trafo.forward(x, *args, **kwargs)
        # avoid rounding errors
        return y.clamp(self._lower_bound, self._upper_bound), dlogp

    def _inverse(self, x, *args, **kwargs):
        return self._trafo.forward(x, *args, **kwargs, inverse=True)
