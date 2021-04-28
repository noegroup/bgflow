import torch
import numpy as np

from .base import Flow

# TODO: write docstrings
# TODO: refactor messy implementation


class FunnelFlow(Flow):
    def __init__(self, eps=1e-6, min_val=-1.0, max_val=1.0):
        super().__init__()
        self._eps = eps
        self._min_val = min_val
        self._max_val = max_val
    
    def _forward(self, x, **kwargs):
        dlogp = (
            torch.nn.functional.logsigmoid(x)
            - torch.nn.functional.softplus(x)
            + np.log(self._max_val - self._min_val)
        ).sum(dim=-1, keepdim=True)
        x = torch.sigmoid(x)
        x = x * (self._max_val - self._min_val) + self._min_val
        x = torch.clamp(x, self._min_val + self._eps, self._max_val - self._eps)
        return x, dlogp
    
    def _inverse(self, x, **kwargs):
        x = torch.clamp(x, self._min_val + self._eps, self._max_val - self._eps)
        x = (x - self._min_val) / (self._max_val - self._min_val)
        dlogp = (
            -torch.log(x - x.pow(2)) 
            - np.log(self._max_val - self._min_val)
        ).sum(dim=-1, keepdim=True)
        x = torch.log(x) - torch.log(1 - x)
        return x, dlogp