
import torch
from .energy import Energy
from .sampling import Sampler


__all__ = ["TorchDistribution", "CustomDistribution"]


class CustomDistribution(Energy, Sampler):
    """Distribution object that is composed of an energy and a sampler.

    Parameters
    ----------
    energy : Energy
        The energy of the distribution.
    sampler : Sampler
        The object that samples from the distribution.

    Notes
    -----
    It is the user's responsibility to ensure that the sampler and energy are consistent.
    """
    def __init__(self, energy, sampler, **kwargs):
        super().__init__(dim=energy.event_shapes, **kwargs)
        self._delegate_energy = energy
        self._delegate_sampler = sampler

    def _energy(self, *args, **kwargs):
        return self._delegate_energy._energy(*args, **kwargs)

    def _sample(self, *args, **kwargs):
        return self._delegate_sampler._sample(*args, **kwargs)

    def _sample_with_temperature(self, *args, **kwargs):
        return self._delegate_sampler._sample_with_temperature(*args, **kwargs)


class TorchDistribution(Energy, Sampler):
    """Wrapper for torch.components.Distribution objects. Instances
    of this class provide all methods and attributes of torch components,
    while also implementing the `Energy` and `Sampler` interfaces in bgtorch.
    """
    def __init__(self, distribution: torch.distributions.Distribution):
        self._delegate = distribution
        super().__init__(dim=distribution.event_shape)

    def _sample(self, n_samples):
        if isinstance(n_samples, int):
            return self._delegate.sample(torch.Size([n_samples]))
        else:
            return self._delegate.sample(torch.Size(n_samples))

    def _sample_with_temperature(self, n_samples, temperature):
        # TODO (implement for distributions that support this feature: Normal, Weibull, ...)
        raise NotImplementedError()

    def _energy(self, x):
        return -self._delegate.log_prob(x)[:,None]

    def __getattr__(self, name):
        try:
            return getattr(self._delegate, name)
        except AttributeError as e:
            msg = str(e)
            msg = msg.replace(self._delegate.__class__.__name__, "Distribution")
            raise AttributeError(msg)
