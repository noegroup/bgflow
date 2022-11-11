__all__ = ["IncreaseMultiplicityFlow", "CircularShiftFlow"]

import torch
from bgflow.nn.flow.transformer.base import Flow


class IncreaseMultiplicityFlow(Flow):
    """A flow that increases the multiplicity of torsional degrees of freedom.
    The input and output tensors are expected to be in [0,1].
    The output represents the sum over sheaves from the input.

    Parameters
    ----------
    multiplicities : Union[torch.Tensor, int]
        A tensor of integers that define the number of periods in the unit interval.
    """

    def __init__(self, multiplicities):
        super().__init__()
        self.register_buffer("_multiplicities", torch.as_tensor(multiplicities))

    def _forward(self, x, **kwargs):
        _assert_in_unit_interval(x)
        multiplicities = torch.ones_like(x) * self._multiplicities
        sheaves = _randint(multiplicities)
        y = (x + sheaves) / self._multiplicities
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp

    def _inverse(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x % (1 / self._multiplicities)) * self._multiplicities
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp


def _randint(high):
    with torch.no_grad():
        return torch.floor(torch.rand(high.shape, device=high.device) * high)


def _assert_in_unit_interval(x):
    if (x > 1 + 1e-6).any() or (x < - 1e-6).any():
        raise ValueError(f'IncreaseMultiplicityFlow operates on [0,1] but input was {x}')


class CircularShiftFlow(Flow):
    """A flow that shifts the position of torsional degrees of freedom.
    The input and output tensors are expected to be in [0,1].
    The output is a translated version of the input, respecting circulariry.

    Parameters
    ----------
    shift : Union[torch.Tensor, float]
        A tensor that defines the translation of the circular interval
    """

    def __init__(self, shift):
        super().__init__()
        self.register_buffer("_shift", torch.as_tensor(shift))

    def _forward(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x + self._shift) % 1
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp

    def _inverse(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x - self._shift) % 1
        dlogp = torch.zeros_like(x[..., [0]])
        return y, dlogp
