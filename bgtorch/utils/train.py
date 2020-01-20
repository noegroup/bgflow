import torch
import numpy as np

from .types import assert_numpy


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
    # cutoff x after b - this should also cutoff infinities
    vals = torch.where(vals < max_val, vals, max_val * torch.ones_like(vals))
    # log after a
    vals_soft = high_val + torch.where(vals < high_val, vals - high_val, torch.log(vals - high_val + 1))
    # make sure everything is finite
    vals_finite = torch.where(torch.isfinite(vals_soft), vals_soft, max_val * torch.ones_like(vals_soft))
    return vals_finite


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


class LossReporter:
    """
        Simple reporter use for reporting losses and plotting them.
    """
    
    def __init__(self, *labels):
        self._labels = labels
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]
    
    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))
    
    def plot(self, n_smooth=10, log=False):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(self._n_reported, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.set_size_inches((8, 4 * self._n_reported), forward=True)
        for i, (label, raw, axis) in enumerate(zip(self._labels, self._raw, axes)):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            smoothed = np.convolve(raw, kernel, mode="valid")
            if not log:
                axis.plot(smoothed)
            else:
                axis.semilogy(smoothed - smoothed.min())
            axis.set_ylabel(label)
            if i == self._n_reported - 1:
                axis.set_xlabel("Iteration")
                
    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])