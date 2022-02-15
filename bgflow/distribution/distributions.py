
import torch
from .energy import Energy
from .sampling import Sampler
from torch.distributions import constraints


__all__ = ["TorchDistribution", "CustomDistribution", "UniformDistribution", "SloppyUniform"]


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
    while also implementing the `Energy` and `Sampler` interfaces in bgflow.
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


class _SloppyUniform(torch.distributions.Uniform):
    def __init__(self, *args, tol=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tol = tol

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low-self.tol, self.high+self.tol)


class SloppyUniform(torch.nn.Module):
    def __init__(self, low, high, validate_args=None, tol=1e-5):
        super().__init__()
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.tol = tol
        self.validate_args = validate_args

    def __getattr__(self, name):
        try:
            return super().__getattr__(name=name)
        except AttributeError:
            uniform = _SloppyUniform(self.low, self.high, self.validate_args, tol=self.tol)
            if hasattr(uniform, name):
                return getattr(uniform, name)
        except:
            raise AttributeError(f"SloppyUniform has no attribute {name}")


class UniformDistribution(TorchDistribution):
    """Shortcut"""
    def __init__(self, low, high, tol=1e-5, validate_args=None, n_event_dims=1):
        uniform = SloppyUniform(low, high, validate_args, tol=tol)
        independent = torch.distributions.Independent(uniform, n_event_dims)
        super().__init__(independent)
        self.uniform = uniform

    def _energy(self, x):
        try:
            y = - self._delegate.log_prob(x)[:,None]
            assert torch.all(torch.isfinite(y))
            return y
        except (ValueError, AssertionError):
            return -self._delegate.log_prob(self._delegate.sample(sample_shape=x.shape[:-1]))[:,None]

    def _sample_with_temperature(self, n_samples, temperature):
        return self._sample(n_samples)
