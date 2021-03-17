
import numpy as np
import torch
from .base import Sampler


__all__ = ["DataSetSampler"]


class DataSetSampler(Sampler):
    """Sample from a data set.

    Parameters
    ----------
    data : torch.Tensor
        The data set from which to draw samples.
    """
    def __init__(self, data: torch.Tensor, ):
        super().__init__()
        self._current_index = 0
        self._idxs = np.random.permutation(len(data))
        self.register_buffer("_data", data)

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        return self._data[idx]

    def _sample(self, n_samples: int, *args, **kwargs):
        if self._current_index + n_samples < len(self._data):
            idxs = self._idxs[self._current_index:self._current_index + n_samples]
            self._current_index += n_samples
            return self._data[idxs]
        else:
            taken = self._data[self._idxs[self._current_index:]]
            np.random.shuffle(self._idxs)
            self._current = 0
            remaining = self._sample(n_samples - len(taken))
            return torch.cat([taken, remaining], dim=0)

