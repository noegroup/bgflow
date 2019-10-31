import torch
import numpy as np


class AffineLayer(torch.nn.Module):
    def __init__(self, n_dims, use_scaling=True, use_translation=True):
        super().__init__()
        self._n_dims = n_dims
        self._log_sigma = None
        if use_scaling:
            self._log_sigma = torch.nn.Parameter(torch.zeros(self._n_dims))
        if use_translation:
            self._mu = torch.nn.Parameter(torch.zeros(self._n_dims))

    def forward(self, x, inverse=False):
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        if not inverse:
            if self._log_sigma is not None:
                sigma = torch.exp(self._log_sigma.to(x))
                dlogp = dlogp + self._log_sigma.sum()
                x = sigma * x
            if self._mu is not None:
                x = x + self._mu
        else:
            if self._mu is not None:
                x = x - self._mu
            if self._log_sigma is not None:
                sigma_inv = torch.exp(-self._log_sigma.to(x))
                dlogp = dlogp - self._log_sigma.sum()
                x = sigma_inv * x
        return x, dlogp
