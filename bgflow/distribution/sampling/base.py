
from typing import Tuple
import torch
from ...utils.types import unpack_tensor_tuple, pack_tensor_in_list

__all__ = ["Sampler"]


class Sampler(torch.nn.Module):
    """Abstract base class for samplers.

    Parameters
    ----------
    return_hook : Callable, optional
        A function to postprocess the samples. This can (for example) be used to
        only return samples at a selected thermodynamic state of a replica exchange sampler
        or to combine the batch and sample dimension.
        The function takes a list of tensors and should return a list of tensors.
        Each tensor contains a batch of samples.
    """

    def __init__(self, return_hook=lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.return_hook = return_hook
    
    def _sample_with_temperature(self, n_samples, temperature, *args, **kwargs):
        raise NotImplementedError()
        
    def _sample(self, n_samples, *args, **kwargs):
        raise NotImplementedError()
    
    def sample(self, n_samples, temperature=1.0, *args, **kwargs):
        """Create a number of samples.

        Parameters
        ----------
        n_samples : int
            The number of samples to be created.
        temperature : float, optional
            The relative temperature at which to create samples.
            Only available for sampler that implement `_sample_with_temperature`.

        Returns
        -------
        samples : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            If this sampler reflects a joint distribution of multiple tensors,
            it returns a tuple of tensors, each of which have length n_samples.
            Otherwise it returns a single tensor of length n_samples.
        """
        if isinstance(temperature, float) and temperature == 1.0:
            samples = self._sample(n_samples, *args, **kwargs)
        else:
            samples = self._sample_with_temperature(n_samples, temperature, *args, **kwargs)
        samples = pack_tensor_in_list(samples)
        return unpack_tensor_tuple(self.return_hook(samples))

    def sample_to_cpu(self, n_samples, batch_size=64, *args,  **kwargs):
        """A utility method for creating many samples that might not fit into GPU memory."""
        with torch.no_grad():
            samples = self.sample(min(n_samples, batch_size), *args, **kwargs)
            samples = pack_tensor_in_list(samples)
            samples = [tensor.detach().cpu() for tensor in samples]
            while len(samples[0]) < n_samples:
                new_samples = self.sample(min(n_samples-len(samples[0]), batch_size), *args, **kwargs)
                new_samples = pack_tensor_in_list(new_samples)
                for i, new in enumerate(new_samples):
                    samples[i] = torch.cat([samples[i], new.detach().cpu()], dim=0)
        return unpack_tensor_tuple(samples)
