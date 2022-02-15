import torch

from .base import Flow


__all__ = ["AffineFlow"]


class AffineFlow(Flow):
    def __init__(self, n_dims, use_scaling=True, use_translation=True):
        super().__init__()
        self._n_dims = n_dims
        self._log_sigma = None
        if use_scaling:
            self._log_sigma = torch.nn.Parameter(torch.zeros(self._n_dims))
        else:
            self._log_sigma = None
        if use_translation:
            self._mu = torch.nn.Parameter(torch.zeros(self._n_dims))
        else:
            self._mu = None

    def _forward(self, x, **kwargs):
        assert x.shape[-1] == self._n_dims, "dimension `x` does not match `n_dims`"
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        if self._log_sigma is not None:
            sigma = torch.exp(self._log_sigma.to(x))
            dlogp = dlogp + self._log_sigma.sum()
            x = sigma * x
        if self._mu is not None:
            x = x + self._mu.to(x)
        return x, dlogp

    def _inverse(self, x, **kwargs):
        assert x.shape[-1] == self._n_dims, "dimension `x` does not match `n_dims`"
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        if self._mu is not None:
            x = x - self._mu.to(x)
        if self._log_sigma is not None:
            sigma_inv = torch.exp(-self._log_sigma.to(x))
            dlogp = dlogp - self._log_sigma.sum()
            x = sigma_inv * x
        return x, dlogp
