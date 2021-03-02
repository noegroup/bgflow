from .base import Flow
from torch.nn.parameter import Parameter
import torch


__all__ = ["ScalingLayer", "EntropyScalingLayer"]


class ScalingLayer(Flow):
    def __init__(self, init_factor=1.0, dim=1):
        super().__init__()
        self._scalefactor = Parameter(init_factor * torch.ones(1))
        self.dim = dim

    def _forward(self, x, *cond, **kwargs):
        n_batch = x.shape[0]
        y = torch.zeros_like(x)
        y[:, : self.dim] = x[:, : self.dim] * self._scalefactor
        y[:, self.dim :] = x[:, self.dim :]
        return (
            y,
            (self.dim * self._scalefactor.log()).repeat(n_batch, 1),
        )

    def _inverse(self, x, *cond, **kwargs):
        n_batch = x.shape[0]
        y = torch.zeros_like(x)
        y[:, : self.dim] = x[:, : self.dim] / self._scalefactor
        y[:, self.dim :] = x[:, self.dim :]
        return (
            y,
            (-self.dim * self._scalefactor.log()).repeat(n_batch, 1),
        )


class EntropyScalingLayer(Flow):
    def __init__(self, init_factor=1.0, dim=1):
        super().__init__()
        self._scalefactor = Parameter(init_factor * torch.ones(1))
        self.dim = dim

    def _forward(self, x, y, *cond, **kwargs):
        n_batch = x.shape[0]
        return (
            self._scalefactor * x,
            y,
            (self.dim * self._scalefactor.log()).repeat(n_batch, 1),
        )

    def _inverse(self, x, y, *cond, **kwargs):
        n_batch = x.shape[0]
        return (
            x / self._scalefactor,
            y,
            (-self.dim * self._scalefactor.log()).repeat(n_batch, 1),
        )
