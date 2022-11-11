import torch
import numpy as np
from typing import Union
from functools import partial


from .types import assert_numpy, unpack_tensor_tuple


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


def linlogcut(vals, high_val=1e3, max_val=1e9):
    cut = torch.where(vals >= high_val, high_val + torch.log(1 + vals - high_val), vals)
    return cut.clamp(min=None, max=max_val)


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


class ClipGradient(torch.nn.Module):
    """A module that clips the gradients in the backward pass.

    Parameters
    ----------
    clip
        the max norm
    norm_dim
        the dimension of the space over which the norm is computed
        - `1` corresponds to clipping by value
        - `3` corresponds to clipping by atom
        - `-1` corresponds to clipping the norm of the whole tensor
    """

    def __init__(self, clip: Union[float, torch.Tensor], norm_dim: int = 1):
        super().__init__()
        self.register_buffer("clip", torch.as_tensor(clip))
        self.norm_dim = norm_dim

    def forward(self, *xs):
        for x in xs:
            if x.requires_grad:
                x.register_hook(partial(ClipGradient.clip_tensor, clip=self.clip, last_dim=self.norm_dim))
        return unpack_tensor_tuple(xs)

    @staticmethod
    def clip_tensor(tensor, clip, last_dim):
        clip.to(tensor)
        original_shape = tensor.shape
        last_dim = (-1, ) if last_dim == -1 else (-1, last_dim)
        out = torch.nan_to_num(tensor, nan=0.0).flatten().reshape(*last_dim)
        norm = torch.linalg.norm(out.detach(), dim=-1, keepdim=True)
        factor = (clip.view(-1, *clip.shape) / norm.view(-1, *clip.shape)).view(-1)
        factor = torch.minimum(factor, torch.ones_like(factor))
        out = out.view(*last_dim) * factor.view(-1, 1)
        out = out.reshape(original_shape)
        return out



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
