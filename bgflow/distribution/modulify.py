import torch
from torch import nn
from torch.distributions.utils import lazy_property


__all__ = ["distribution_module"]


def distribution_module(cls):
    """A class decorator to "modulify" torch distributions."""
    # get the actual distribution class from the mro()
    cls_distribution = []
    for parent in cls.mro()[1:]:
        if (
            issubclass(parent, torch.distributions.Distribution)
            and parent != torch.distributions.Distribution
        ):
            cls_distribution.append(parent)
    assert len(cls_distribution) == 1
    cls_distribution = cls_distribution[0]

    class DistributionModule:
        def __init__(self, *args, **kwargs):
            """Initialize distribution and register plain tensors as buffers."""
            super().__init__(*args, **kwargs)

            # after initializing re-register distribution parameters as buffers
            k_tensors = []
            for k in self.__dict__:

                # check if it's a lazy_property
                # in this case we should simply move on instead of evaluating it.
                if self._is_lazy_property(k):
                    continue

                # check if it's a "plain" tensor
                if isinstance(getattr(self, k), torch.Tensor):
                    k_tensors.append(k)

            for k in k_tensors:
                # for a "plain "tensor we will now register it as buffer.
                val = getattr(self, k)
                delattr(self, k)
                self.register_buffer(k, val)

        def __getattribute__(self, name):
            """Return attribute with lazy_property special check."""
            if type(self)._is_lazy_property(name):
                # deleting the attribute from the instance will simply "reset"
                # the lazy_property
                delattr(self, name)
            return super().__getattribute__(name)

        @classmethod
        def _is_lazy_property(cls, name):
            return isinstance(getattr(cls, name, object), lazy_property)

    return type(
        cls.__name__, (DistributionModule, cls_distribution, nn.Module), {}
    )