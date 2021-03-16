
import torch
from .energy import Energy
from .sampling import Sampler


__all__ = ["Distribution"]


class Distribution(Energy, Sampler):
    """Wrapper for torch.distributions.Distribution objects. Instances
    of this class provide all methods and attributes of torch distributions,
    while also implementing the `Energy` and `Sampler` interfaces in bgtorch.
    """
    def __init__(self, distribution: torch.distributions.Distribution):
        self._delegate = distribution
        Energy.__init__(self, dim=distribution.event_shape)
        Sampler.__init__(self)

    def _sample(self, n_samples):
        if isinstance(n_samples, int):
            return self._delegate.sample(torch.Size([n_samples]))
        else:
            return self._delegate.sample(torch.Size(n_samples))

    def _energy(self, x):
        return -self._delegate.log_prob(x)[:,None]

    def __getattr__(self, name):
        try:
            return getattr(self._delegate, name)
        except AttributeError as e:
            msg = str(e)
            msg = msg.replace(self._delegate.__class__.__name__, "Distribution")
            raise AttributeError(msg)
