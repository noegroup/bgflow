import warnings

import torch
import numpy as np

from .base import Flow
from .inverted import InverseFlow

# TODO: write docstrings


class SplitFlow(Flow):
    def __init__(self, n_left=1):
        super().__init__()
        self._n_left = n_left

    def _forward(self, x, *cond, **kwargs):
        assert x.shape[-1] > self._n_left, "input dim must be larger than `n_left`"
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        x_left = x[..., : self._n_left]
        x_right = x[..., self._n_left :]
        return (x_left, x_right, *cond, dlogp)
    
    def _inverse(self, *xs, **kwargs):
        cond = xs[2:]
        xs = xs[:2]
        x_left, x_right = xs
        dlogp = torch.zeros(*x_left.shape[:-1], 1).to(x_left)
        x = torch.cat([x_left, x_right], dim=-1)
        return (x, *cond, dlogp)


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
