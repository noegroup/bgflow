import torch
from ...utils.types import unpack_tensor_tuple, pack_tensor_in_list

__all__ = ["Sampler"]


class Sampler(torch.nn.Module):
    """Abstract base class for samplers.

    Parameters
    ----------
    select_fn : Callable, optional
        A function to postprocess the samples. This can (for example) be used to
        only return samples at a selected thermodynamic state of a replica exchange sampler.
    device : torch.device.device
        The device on which the sampled tensors should live.
    dtype : torch.dtype
        Data type of the sampled tensors.
    """

    def __init__(self, select_fn=lambda x: x, device=None, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.select_fn = select_fn
        # context dummy tensor to store the device and data type;
        # by specifying this here, each subclass can decide for itself
        # whether it wants to store the data on the gpu or cpu.
        self.register_buffer("ctx", torch.tensor([], device=device, dtype=dtype))
    
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
        samples : Union[torch.Tensor, tuple[torch.Tensor]]
            If this sampler reflects a joint distribution of multiple tensors,
            it returns a tuple of tensors, each of which have length n_samples.
            Otherwise it returns a single tensor of length n_samples.
        """
        if temperature != 1.0:
            samples = self._sample_with_temperature(n_samples, temperature, *args, **kwargs)
        else:
            samples = self._sample(n_samples, *args, **kwargs)
        samples = pack_tensor_in_list(samples)
        samples = [sample.to(self.ctx) for sample in samples]
        return unpack_tensor_tuple(self.select_fn(samples))

    def sample_to_cpu(self, n_samples, *args, batch_size=64, **kwargs):
        """A utility method for creating many samples that might not fit into GPU memory."""
        with torch.no_grad():
            samples = self.sample(min(n_samples, batch_size), *args, **kwargs)
            samples = pack_tensor_in_list(samples)
            samples = [tensor.detach().cpu() for tensor in samples]
            while len(samples[0]) < n_samples:
                new_samples = self.sample(min(n_samples-len(samples[0]), batch_size), *args, **kwargs)
                new_samples = pack_tensor_in_list(new_samples)
                for i, new in enumerate(new_samples):
                    samples[i] = torch.cat([samples[i], new], dim=0)
        return unpack_tensor_tuple(samples)
