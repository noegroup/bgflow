import numpy as np
import torch
from itertools import product

from .base import Flow


def _make_checkerboard_idxs(sz):
    even = np.arange(sz, dtype=np.int64) % 2 
    odd = 1 - even
    grid = np.arange(sz * sz, dtype=np.int64)
    idxs = []
    for i, j in product([odd, even], repeat=2):
        mask = np.outer(i, j).astype(bool).reshape(-1)
        chunk = grid[mask]
        idxs.append(chunk)
    return np.concatenate(idxs)


def _checkerboard_2x2_masks(sz):
        mask = _make_checkerboard_idxs(sz)
        inv_mask = np.argsort(mask)
        offset = sz ** 2 // 4
        sub_masks = [
            mask[i * offset:(i+1) * offset] 
            for i in range(4)
        ]
        return inv_mask, sub_masks


class CheckerboardFlow(Flow):
    def __init__(self, size):
        super().__init__()
        self._size = size
        inv_mask, submasks = _checkerboard_2x2_masks(size)
        self.register_buffer("_sub_masks", torch.LongTensor(submasks))
        self.register_buffer("_inv_mask", torch.LongTensor(inv_mask))
    
    def _forward(self, *xs, **kwargs):
        n_batch = xs[0].shape[0]
        dlogp = torch.zeros(n_batch)
        sz = self._size // 2
        assert len(xs) == 1
        x = xs[0]
        assert len(x.shape) == 4 and x.shape[1] == self._size and x.shape[2] == self._size,\
            "`x` needs to be of shape `[n_batch, size, size, n_filters]`"
        x = x.view(n_batch, self._size ** 2, -1)
        xs = []
        for i in range(4):
            patch = x[:, self._sub_masks[i], :].view(n_batch, sz, sz, -1)
            xs.append(patch)
        return (*xs, dlogp)
        return x, dlogp
    
    def _inverse(self, *xs, **kwargs):
        n_batch = xs[0].shape[0]
        dlogp = torch.zeros(n_batch)
        sz = self._size // 2
        assert len(xs) == 4
        assert all(x.shape[1] == self._size // 2 and x.shape[2] == self._size // 2 for x in xs),\
            "all `xs` needs to be of shape `[n_batch, size, size, n_filters]`"
        xs = [x.view(n_batch, sz ** 2, -1) for x in xs]
        x = torch.cat(xs, axis=-2)[:, self._inv_mask, :].view(
            n_batch, self._size, self._size, -1
        )
        return x, dlogp
