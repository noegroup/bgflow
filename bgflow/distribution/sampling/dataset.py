
import numpy as np
import torch
from .base import Sampler
from ...utils.types import unpack_tensor_tuple, pack_tensor_in_list


__all__ = ["DataLoaderSampler", "DataSetSampler"]


class _ToDeviceSampler(Sampler):
    """A sampler that can move data between devices and data types"""
    def __init__(self, device, dtype, return_hook=lambda x: x, **kwargs):
        super().__init__(return_hook=return_hook, **kwargs)
        # context dummy tensor to store the device and data type;
        # by specifying this here, the user can decide
        # if he/she wants to store the data on the gpu or cpu.
        self.register_buffer("ctx", torch.tensor([], device=device, dtype=dtype))

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        samples = pack_tensor_in_list(samples)
        samples = [s.to(self.ctx) for s in samples]
        return unpack_tensor_tuple(samples)


class DataLoaderSampler(_ToDeviceSampler):
    """A torch.DataLoader instance wrapped as a sampler.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The data loader instance.
    device : torch.device.device
        The device on which the sampled tensors should live.
    dtype : torch.dtype
        Data type of the sampled tensors.

    Notes
    -----
    Only implemented for dataloader.batch_size == n_samples
    """
    def __init__(self, dataloader, device, dtype, return_hook=lambda x: x):
        super().__init__(device=device, dtype=dtype, return_hook=return_hook)
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)

    def _sample(self, n_samples, *args, **kwargs):
        if n_samples != self._dataloader.batch_size:
            raise ValueError("DataLoaderSampler only implemented for batch_size == n_samples")
        samples = next(self._iterator)
        return unpack_tensor_tuple(samples)


class DataSetSampler(_ToDeviceSampler, torch.utils.data.Dataset):
    """Sample from data.

    Parameters
    ----------
    *data : torch.Tensor
        Potentially multiple torch tensors of the same length.
    shuffle : bool
        Whether the data should be accessed in random order.
    device : torch.device.device
        The device that the sampled tensors should live on.
    dtype : torch.dtype
        Data type of the sampled tensors.

    Attributes
    ----------
    data : list[torch.Tensor]
        The data set from which to draw samples.
    """
    def __init__(self, *data: torch.Tensor, shuffle=True, device=None, dtype=None, return_hook=lambda x: x):
        device = data[0].device if device is None else device
        dtype = data[0].dtype if dtype is None else dtype
        super().__init__(device=device, dtype=dtype, return_hook=return_hook)
        if not all(len(d) == len(data[0]) for d in data):
            raise ValueError("All data items must have the same length.")

        self.data = pack_tensor_in_list(data)
        self._current_index = 0
        self._shuffle = shuffle
        if shuffle:
            self._idxs = np.random.permutation(len(data[0]))
        else:
            self._idxs = np.arange(len(data[0]))

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

    def _sample(self, n_samples: int, *args, **kwargs):
        samples = [torch.tensor([]).to(self.data[0]) for _ in self.data]
        if self._current_index + n_samples < len(self.data[0]):
            idxs = self._idxs[self._current_index:self._current_index + n_samples]
            self._current_index += n_samples
            for i in range(len(self.data)):
                samples[i] = torch.cat([samples[i], self.data[i][idxs]], dim=0)
        else:
            idxs = self._idxs[self._current_index:]
            for i in range(len(self.data)):
                samples[i] = torch.cat([samples[i], self.data[i][idxs]], dim=0)
            # reset
            if self._shuffle:
                np.random.shuffle(self._idxs)
            self._current_index = 0
            # recurse
            remaining = self._sample(n_samples - len(samples[0]))
            remaining = pack_tensor_in_list(remaining)
            for i, other in enumerate(remaining):
                samples[i] = torch.cat([samples[i], other], dim=0)
        return unpack_tensor_tuple(samples)

    def reshuffle_(self):
        """Shuffle the dataset randomly in-place."""
        self._idxs = np.random.permutation(len(self.data[0]))
        self._current_index = 0
        return self

    def resize_(self, new_size):
        """Resize the data set to `new_size` and reinitialize the randomization.
        - When resizing to a bigger size, samples from the set are randomly repeated.
        - When resizing to a smaller size, samples from the set are randomly deleted.

        Returns
        -------
        indices : np.ndarray
            The indices used for reshuffling.

        Notes
        -----
        This is an in-place operation.
        """
        if new_size != len(self):
            indices = np.random.randint(low=0, high=len(self), size=new_size)
            for i in range(len(self.data)):
                self.data[i] = self.data[i][indices]
            self._idxs = np.random.permutation(new_size)
            self._current_index = 0
            return indices
        else:
            return np.arange(len(self))
