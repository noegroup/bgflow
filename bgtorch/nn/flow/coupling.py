import warnings

import torch
import numpy as np

from .base import Flow
from .inverted import InverseFlow

# TODO: write docstrings

__all__ = ["SplitFlow", "MergeFlow", "SwapFlow", "CouplingFlow"]


class SplitFlow(Flow):
    """Split the input tensor into multiple output tensors.

    Parameters
    ----------
    *sizes : int
        Lengths of the output tensors in dimension `dim`.
    dim : int
        Dimension along which to split.

    Raises
    ------
    ValueError
        If the tensor is to short for the desired split in dimension `dim`.

    Notes
    -----
    Specifying the length of the last tensor is optional. If the tensor is longer
    than the sum of all sizes, the last size will be inferred from the input
    dimensions.
    """
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
        return (*y, self._dlogp(x))
    
    def _inverse(self, *xs, **kwargs):
        y = torch.cat(xs, dim=self._split_dim)
        shape = list(xs[0].shape)
        shape[self._split_dim] = 1
        return y, self._dlogp(xs[0])

    def _dlogp(self, x):
        index = [slice(None)] * len(x.shape)
        index[self._split_dim] = slice(1)
        return torch.zeros_like(x[index])


class MergeFlow(InverseFlow):
    def __init__(self, *sizes):
        """ Shortcut to InvertedFlow(SplitFlow()) """
        super().__init__(SplitFlow(*sizes))


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
