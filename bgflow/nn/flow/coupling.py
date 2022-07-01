import warnings
from typing import Sequence

import numpy as np
import torch

from .base import Flow
from .inverted import InverseFlow

__all__ = ["SplitFlow", "MergeFlow", "SwapFlow", "CouplingFlow", "WrapFlow", "SetConstantFlow"]


class SplitFlow(Flow):
    """Split the input tensor into multiple output tensors.

    Parameters
    ----------
    *sizes_or_indices : int or sequence of ints
        If int: lengths of the output tensors in dimension `dim`.
        Otherwise: indices of the input tensor that are mapped each output tensor.

    dim : int
        Dimension along which to split.

    Raises
    ------
    ValueError
        If the tensor is to short for the desired split in dimension `dim`.

    Notes
    -----
    Specifying the size or indices of the last tensor is optional. If the tensor is longer
    than the sum of all sizes, the last size will be inferred from the input
    dimensions.
    """
    def __init__(self, *sizes_or_indices, dim=-1):
        super().__init__()
        if isinstance(sizes_or_indices[0], Sequence) or isinstance(sizes_or_indices[0], np.ndarray):
            self._sizes = None
            self._indices = sizes_or_indices
        else:
            self._sizes = sizes_or_indices
            self._indices = None
        self._split_dim = dim

    def _forward(self, x, **kwargs):
        if self._indices is None:
            return (*self._split_with_sizes(x), self._dlogp(x))
        else:
            return (*self._split_with_indices(x), self._dlogp(x))

    def _inverse(self, *xs, **kwargs):
        if self._indices is None:
            y = torch.cat(xs, dim=self._split_dim)
        else:
            y = self._cat_with_indices(*xs)
        return y, self._dlogp(xs[0])

    def _dlogp(self, x):
        index = [slice(None)] * len(x.shape)
        index[self._split_dim] = slice(1)
        return torch.zeros_like(x[index])

    def _split_with_sizes(self, x):
        last_size = x.shape[self._split_dim] - sum(self._sizes)
        if last_size == 0:
            sizes = self._sizes
        elif last_size > 0:
            sizes = [*self._sizes, last_size]
        else:
            raise ValueError(f"can't split x [{x.shape}] into sizes {self._sizes} along {self._split_dim}")
        return torch.split(x, sizes, dim=self._split_dim)

    def _split_with_indices(self, x):
        is_done = torch.zeros(x.shape[self._split_dim], dtype=torch.bool)
        result = []
        for indices in self._indices:
            if is_done[indices].any():
                raise ValueError("Cannot split tensor. Indices are overlapping.")
            result.append(x[self._range(indices, len(x.shape))])
            is_done[indices] = True
        if not is_done.all():
            raise ValueError(f"Split with indices missed indices {torch.arange(len(is_done))[is_done]}")
        return result

    def _range(self, indices, n_dimensions):
        dims = [slice(None) for _ in range(n_dimensions)]
        dims[self._split_dim] = list(indices)
        return dims

    def _cat_with_indices(self, *xs):
        length = sum(len(indices) for indices in self._indices)
        output_shape = list(xs[0].shape)
        output_shape[self._split_dim] = length
        y = torch.empty(*output_shape, device=xs[0].device, dtype=xs[0].dtype)
        is_done = torch.zeros(length, dtype=torch.bool)
        for x, indices in zip(xs, self._indices):
            if is_done[indices].any():
                raise ValueError("Cannot merge tensor. Indices are overlapping.")
            y[self._range(indices, len(x.shape))] = x
            is_done[indices] = True
        if not is_done.all():
            raise ValueError(f"Merge with indices missed indices {torch.arange(len(is_done))[is_done]}")
        return y


class MergeFlow(InverseFlow):
    def __init__(self, *sizes, dim=-1):
        """ Shortcut to InverseFlow(SplitFlow()) """
        super().__init__(SplitFlow(*sizes, dim=dim))


class SwapFlow(Flow):
    def __init__(self):
        """ Swaps two input channels """
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
    """Coupling Layer

    Parameters
    ----------
    transformer : torch.nn.Module
        the transformer
    transformed_indices : Iterable of int
        indices of the inputs to be transformed
    cond_indices : Iterable of int
        indices of the inputs for the conditioner
    cat_dim : int
        the dimension along which the conditioner inputs are concatenated

    Raises
    ------
    ValueError
        If transformer and conditioner indices are not disjointed.
    """
    def __init__(self, transformer, transformed_indices=(1,), cond_indices=(0,), cat_dim=-1):
        super().__init__()
        self.transformer = transformer
        self.transformed_indices = transformed_indices
        self.cond_indices = cond_indices
        invalid = np.intersect1d(self.transformed_indices, self.cond_indices)
        if len(invalid) > 0:
            raise ValueError(f"Indices {invalid} cannot be both transformed and conditioned on.")
        self.cat_dim = cat_dim

    def _forward(self, *x, **kwargs):
        input_lengths = [x[i].shape[self.cat_dim] for i in self.transformed_indices]
        inputs = torch.cat([x[i] for i in self.transformed_indices], dim=self.cat_dim)
        cond_inputs = torch.cat([x[i] for i in self.cond_indices], dim=self.cat_dim)
        x = list(x)
        y, dlogp = self.transformer.forward(cond_inputs, inputs, **kwargs)
        y = torch.split(y, input_lengths, self.cat_dim)
        for i, yi in zip(self.transformed_indices, y):
            x[i] = yi
        return (*x, dlogp)

    def _inverse(self, *x, **kwargs):
        input_lengths = [x[i].shape[self.cat_dim] for i in self.transformed_indices]
        inputs = torch.cat([x[i] for i in self.transformed_indices], dim=self.cat_dim)
        cond_inputs = torch.cat([x[i] for i in self.cond_indices], dim=self.cat_dim)
        x = list(x)
        y, dlogp = self.transformer.forward(cond_inputs, inputs, **kwargs, inverse=True)
        y = torch.split(y, input_lengths, self.cat_dim)
        for i, yi in zip(self.transformed_indices, y):
            x[i] = yi
        return (*x, dlogp)


class WrapFlow(Flow):
    """Apply a flow to a subset of inputs.

    Parameters
    ----------
    flow : bgflow.Flow
        The flow that is applied to a subset of inputs.
    indices : Iterable of int
        Indices of the inputs that are passed to the `flow`.
    out_indices : Iterable of int
        The outputs of the `flow` are assigned to those outputs of the wrapped flow.
        By default, the out indices are the same as the indices.
    """
    def __init__(self, flow, indices, out_indices=None):
        super().__init__()
        self._flow = flow
        self._indices = indices
        self._argsort_indices = np.argsort(indices)
        self._out_indices = indices if out_indices is None else out_indices
        self._argsort_out_indices = np.argsort(self._out_indices)

    def _forward(self, *xs, **kwargs):
        inp = (xs[i] for i in self._indices)
        output = [xs[i] for i in range(len(xs)) if i not in self._indices]
        *yi, dlogp = self._flow(*inp, **kwargs)
        for i in self._argsort_out_indices:
            index = self._out_indices[i]
            output.insert(index, yi[i])
        return (*tuple(output), dlogp)

    def _inverse(self, *xs, **kwargs):
        inp = (xs[i] for i in self._out_indices)
        output = [xs[i] for i in range(len(xs)) if i not in self._out_indices]
        *yi, dlogp = self._flow(*inp, inverse=True, **kwargs)
        for i in self._argsort_indices:
            index = self._indices[i]
            output.insert(index, yi[i])
        return (*tuple(output), dlogp)




class SetConstantFlow(Flow):
    """A flow that sets some inputs constant in the forward direction and removes them in the inverse.

    Parameters
    ----------
    indices : Sequence[int]
        Indices to be set to constants.
    values : Sequence[tensor]
        Constant values; sequence has to have the same length as `indices`.
    n_event_dims0 : int, optional
        The number of event dims of x[0]. Required to infer the batch shape.
    """

    def __init__(self, indices, values, n_event_dims0=1):
        super().__init__()
        argsort = np.argsort(indices)
        self.indices = [indices[i] for i in argsort]
        values = [values[i] for i in argsort]
        for i, v in enumerate(values):
            self.register_buffer(f"_values_{i}", v)
        self.n_event_dims0 = n_event_dims0

    @property
    def values(self):
        result = []
        i = 0
        while hasattr(self, f"_values_{i}"):
            result.append(getattr(self, f"_values_{i}"))
            i += 1
        return result

    def _forward(self, *xs, **kwargs):
        """insert constants"""
        batch_shape = list(xs[0].shape[:self.n_event_dims0])
        y = list(xs)
        for i, v in zip(self.indices, self.values):
            y.insert(i, v.repeat([*batch_shape, *np.ones_like(v.shape)]))
        dlogp = torch.zeros(batch_shape + [1], device=xs[0].device, dtype=xs[0].dtype)
        return (*y, dlogp)

    def _inverse(self, *xs, **kwargs):
        """remove constants"""
        y = tuple(xs[i] for i, z in enumerate(xs) if i not in self.indices)
        batch_shape = list(y[0].shape[:self.n_event_dims0])
        dlogp = torch.zeros(batch_shape + [1], device=y[0].device, dtype=y[0].dtype)
        return (*y, dlogp)

