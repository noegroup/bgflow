
from typing import Union

import warnings
import torch
import numpy as np

from .base import Transformer

# TODO: write docstring

__all__ = ["AffineTransformer"]


class AffineTransformer(Transformer):
    """RealNVP/NICE

    Parameters
    ----------
    is_circular : bool
        Whether this transform is periodic on [0,1].
    """
    def __init__(
        self,
        shift_transformation=None,
        scale_transformation=None,
        init_downscale=1.0,
        preserve_volume=False,
        is_circular=False,
    ):
        if scale_transformation is not None and is_circular:
            raise ValueError("Scaling is not compatible with periodicity.")
        super().__init__()
        self._shift_transformation = shift_transformation
        self._scale_transformation = scale_transformation
        self._log_alpha = torch.nn.Parameter(torch.zeros(1) - init_downscale)
        self._preserve_volume = preserve_volume
        self._is_circular = is_circular

    def _get_mu_and_log_sigma(self, x, y, *cond, target_dlogp: Union[float, torch.Tensor] = None):
        if self._shift_transformation is not None:
            mu = self._shift_transformation(x, *cond)
        else:
            mu = torch.zeros_like(y).to(x)
        if self._scale_transformation is not None:
            alpha = torch.exp(self._log_alpha.to(x))
            log_sigma = torch.tanh(self._scale_transformation(x, *cond))
            log_sigma = log_sigma * alpha
            if self._preserve_volume:
                target_dlogp = 0.0 if target_dlogp is None else target_dlogp
                target_scale = target_dlogp / np.prod(log_sigma[0].shape)
                log_sigma = (
                        log_sigma
                        - log_sigma.mean(dim=-1, keepdim=True)
                        + target_scale * torch.ones_like(log_sigma)
                )
            else:
                if target_dlogp is not None:
                    warnings.warn("target_dlogp is only effective is self.preserve_volume is enabled.")
        else:
            log_sigma = torch.zeros_like(y).to(x)
        return mu, log_sigma

    def _forward(self, x, y, *cond, target_dlogp=None, **kwargs):
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond, target_dlogp=target_dlogp)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma = torch.exp(log_sigma)
        dlogp = (log_sigma).sum(dim=-1, keepdim=True)
        y = sigma * y + mu
        if self._is_circular:
            y = y % 1.0
        return y, dlogp

    def _inverse(self, x, y, *cond, target_dlogp=None, **kwargs):
        mu, log_sigma = self._get_mu_and_log_sigma(x, y, *cond, target_dlogp=None if target_dlogp is None else -target_dlogp)
        assert mu.shape[-1] == y.shape[-1]
        assert log_sigma.shape[-1] == y.shape[-1]
        sigma_inv = torch.exp(-log_sigma)
        dlogp = (-log_sigma).sum(dim=-1, keepdim=True)
        y = sigma_inv * (y - mu)
        if self._is_circular:
            y = y % 1.0
        return y, dlogp
