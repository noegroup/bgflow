import warnings

import torch
import numpy as np

from .base import Flow
from .inverted import InverseFlow

# TODO: write docstrings

__all__ = ["SplitFlow", "MergeFlow", "SwapFlow", "CouplingFlow"]


class SplitFlow(Flow):
    def __init__(self, *sizes, dim=-1):
        super().__init__()
        self._sizes = sizes
        self._split_dim = dim
    
    def _forward(self, x, **kwargs):
        last_size = x.shape[self._split_dim] - sum(self._sizes)
        if last_size == 0:
            sizes = self._sizes
        elif last_size > 0:
            sizes = [*self._sizes, last_size]
        else:
            raise ValueError(f"can't split x [{x.shape}] into sizes {self._sizes} along {self._split_dim}")
        *y, = torch.split(x, sizes, dim=self._split_dim)
        dlogp = torch.zeros_like(x[...,[0]])
        return (*y, dlogp)
    
    def _inverse(self, *xs, **kwargs):
        y = torch.cat(xs, dim=self._split_dim)
        dlogp = torch.zeros_like(xs[0][...,[0]])
        return (y, dlogp)


class MergeFlow(InverseFlow):
    def __init__(self, n_left=1):
        """ Shortcut to InvertedFlow(SplitFlow()) """
        super().__init__(SplitFlow(n_left=n_left))


class SwapFlow(Flow):
    def __init__(self):
        """ Swaps input channels """
        super().__init__()
        
    def _forward(self, *xs, **kwargs):
        dlogp = torch.zeros(*xs[0].shape[:-1], 1).to(xs[0])
        if len(xs) == 1:
            warnings.warn("applying swapping on a single tensor has no effect")
        xs = (xs[1], xs[0], *xs[2:])
        return (*xs, dlogp)

    def _inverse(self, *xs, **kwargs):
        dlogp = torch.zeros(*xs[0].shape[:-1], 1).to(xs[0])
        if len(xs) == 1:
            warnings.warn("applying swapping on a single tensor has no effect")
        xs = (xs[1], xs[0], *xs[2:])
        return (*xs, dlogp)


class CouplingFlow(Flow):
    def __init__(self, transformer, dt=1.0):
        super().__init__()
        self._transformer = transformer
        assert dt > 0
        self._dt = dt

    def _forward(self, x_left, x_right, *cond, **kwargs):
        x_right, dlogp = self._transformer._forward(x_left, x_right, *cond, **kwargs)
        return x_left, x_right, dlogp
    
    def _inverse(self, x_left, x_right, *cond, **kwargs):
        x_right, dlogp = self._transformer._inverse(x_left, x_right, *cond, **kwargs)
        return x_left, x_right, dlogp
