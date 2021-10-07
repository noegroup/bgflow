import torch

from .affine import AffineTransformer

__all__ =  ["TruncatedGaussianTransformer"]


class TruncatedGaussianTransformer(AffineTransformer):
    """Similar to CDFTransform(TruncatedNormalDistribution()), but with conditioning of mu and sigma.

    Parameters
    ----------
    mu_transformation : torch.nn.Module
        A network to compute the mu of the CDF transform; see `AffineTransformer.shift_net`
    mu_transformation : torch.nn.Module
        A network to compute the sigma of the CDF transform; see `AffineTransformer.scale_net`
    lower_bound_in : Union[torch.Tensor, np.ndarray, float]
        lower bound of the input
    upper_bound_in : Union[torch.Tensor, np.ndarray, float]
        upper bound of the input
    lower_bound_out : Union[torch.Tensor, np.ndarray, float]
        lower bound of the output
    upper_bound_out : Union[torch.Tensor, np.ndarray, float]
        upper bound of the output
    """
    def __init__(
        self,
        mu_transformation=None,
        sigma_transformation=None,
        lower_bound_in=0.0,
        upper_bound_in=1.0,
        lower_bound_out=0.0,
        upper_bound_out=1.0,
    ):
        super().__init__(
            shift_transformation=mu_transformation,
            scale_transformation=sigma_transformation,
        )
        self.register_buffer("_lower_bound_in", torch.tensor(lower_bound_in))
        self.register_buffer("_upper_bound_in", torch.tensor(upper_bound_in))
        self.register_buffer("_lower_bound_out", torch.tensor(lower_bound_out))
        self.register_buffer("_upper_bound_out", torch.tensor(upper_bound_out))

    def _truncated_normal_cdf_log_prob(self, y, mu, sigma, inverse=False):
        """Returns cdf and log_prob"""
        ctx = {"device": mu.device, "dtype": mu.dtype}
        standard_normal = torch.distributions.normal.Normal(
            torch.tensor(0.0, **ctx),
            torch.tensor(1.0, **ctx)
        )  # this is a lightweight class; its construction should not affect performance too much
        alpha = (self._lower_bound_in - mu) / sigma
        beta = (self._upper_bound_in - mu) / sigma
        cdf_lower, cdf_upper = standard_normal.cdf(alpha.detach()), standard_normal.cdf(beta.detach())
        z = cdf_upper - cdf_lower
        if inverse:
            y = standard_normal.icdf(z * y + cdf_lower) * sigma + mu
            log_prob = standard_normal.log_prob((y - mu) / sigma) - torch.log(z * sigma)
            return y, -log_prob
        else:
            log_prob = standard_normal.log_prob((y - mu) / sigma) - torch.log(z * sigma)
            y = (standard_normal.cdf((y - mu) / sigma) - cdf_lower) / z
            return y, log_prob

    def _scale(self, y, lower, upper, inverse=False):
        if inverse:
            y = (y - lower) / (upper - lower)
            dlogp = - torch.log(upper - lower)
        else:
            y = lower + y * (upper - lower)
            dlogp = torch.log(upper - lower)
        return y, dlogp

    def _assert_range(self, y, lower, upper, tol=1e-7, clamp=True):
        assert (y > lower - tol).all()
        assert (y < upper + tol).all()
        if clamp:
            return y.clip(lower, upper)
        else:
            return y

    def _forward(self, x, y, *cond, **kwargs):
        y = self._assert_range(y, self._lower_bound_in, self._upper_bound_in)
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma = torch.exp(log_sigma)
        y, dlogp = self._truncated_normal_cdf_log_prob(y, mu, sigma, inverse=False)
        # y is in [0,1]
        y, dlogp_scale = self._scale(y, lower=self._lower_bound_out, upper=self._upper_bound_out)
        y = self._assert_range(y, self._lower_bound_out, self._upper_bound_out)
        return y, (dlogp + dlogp_scale).sum(dim=-1, keepdim=True)

    def _inverse(self, x, y, *cond, **kwargs):
        y = self._assert_range(y, self._lower_bound_out, self._upper_bound_out)
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma = torch.exp(log_sigma)
        y, dlogp_scale = self._scale(y, lower=self._lower_bound_out, upper=self._upper_bound_out, inverse=True)
        # y is in [0,1]
        y, dlogp = self._truncated_normal_cdf_log_prob(y, mu, sigma, inverse=True)
        y = self._assert_range(y, self._lower_bound_in, self._upper_bound_in)
        return y, (dlogp + dlogp_scale).sum(dim=-1, keepdim=True)