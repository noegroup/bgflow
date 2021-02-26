import torch

from .base import Transformer

# TODO: write docstring


class AffineTransformer(Transformer):
    def __init__(
        self,
        shift_transformation=None,
        scale_transformation=None,
        dt=1.0,
        init_downscale=1.0,
        preserve_volume=False,
    ):
        super().__init__()
        self._shift_transformation = shift_transformation
        self._scale_transformation = scale_transformation
        self._log_alpha = torch.nn.Parameter(torch.zeros(1) - init_downscale)
        self._preserve_volume = preserve_volume
        assert dt > 0
        self._dt = dt

    def _get_mu_and_log_sigma(self, x, y, *cond):
        if self._shift_transformation is not None:
            mu = self._shift_transformation(x, *cond)
        else:
            mu = torch.zeros_like(y).to(x)
        if self._scale_transformation is not None:
            alpha = torch.exp(self._log_alpha.to(x))
            log_sigma = torch.tanh(self._scale_transformation(x, *cond))
            log_sigma = log_sigma * alpha
            if self._preserve_volume:
                log_sigma = log_sigma - log_sigma.mean(dim=-1, keepdim=True)
        else:
            log_sigma = torch.zeros_like(y).to(x)
        return mu, log_sigma

    def _forward(self, x, y, *cond, **kwargs):
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma = torch.exp(log_sigma)
        dlogp = (log_sigma).sum(dim=-1, keepdim=True)
        y = sigma * y + mu
        return y, dlogp

    def _inverse(self, x, y, *cond, **kwargs):
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma_inv = torch.exp(-log_sigma)
        dlogp = (-log_sigma).sum(dim=-1, keepdim=True)
        y = sigma_inv * (y - mu)
        return y, dlogp
