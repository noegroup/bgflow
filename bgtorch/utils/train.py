import torch
import numpy as np


class IndexBatchIterator(object):
    def __init__(self, n_elems, n_batch):
        """
            Produces batches of length `n_batch` of an index set
            `[1, ..., n_elems]` which are sampled randomly without
            replacement.

            If `n_elems` is not a multiple of `n_batch` the last sampled
            batch will be truncated.

            After the iteration throw `StopIteration` its random seed
            will be reset.

            Parameters:
            -----------
            n_elems : Integer
                Number of elements in the index set.
            n_batch : Integer
                Number of batch elements sampled.

        """
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= self._n_elems:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def __len__(self):
        return self._n_elems // self._n_batch

    def next(self):
        return self.__next__()


def linlogcut(vals, high_val=1e3, max_val=1e9, inplace=False):
    if not inplace:
        vals = vals.clone()
    filt = (vals >= high_val)
    diff = vals[filt] - high_val
    vals[filt] = torch.min(
        high_val + torch.log(1 + diff),
        max_val * torch.ones_like(diff)
    )
    return vals


class _ClipGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, max_norm):
        ctx._max_norm = max_norm
        return input

    @staticmethod
    def backward(ctx, grad_output):
        max_norm = ctx._max_norm
        grad_norm = torch.norm(grad_output, p=2, dim=1)
        coeff = max_norm / torch.max(grad_norm, max_norm * torch.ones_like(grad_norm))
        return grad_output * coeff.view(-1, 1), None, None

clip_grad = _ClipGradient.apply