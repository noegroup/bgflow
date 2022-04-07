__all__ = ["IncreaseMultiplicityFlow"]

import torch
from typing import Union
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
        dlogp = dlogp + torch.log(self._multiplicities)
        return y, dlogp

    def _inverse(self, x, **kwargs):
        _assert_in_unit_interval(x)
        y = (x % (1 / self._multiplicities)) * self._multiplicities
        dlogp = torch.zeros_like(x[..., [0]])
        dlogp = dlogp - torch.log(self._multiplicities)
        return y, dlogp


def _randint(high):
    with torch.no_grad():
        return torch.floor(torch.rand(high.shape, device=high.device) * high)


def _assert_in_unit_interval(x):
    if (x > 1 + 1e-6).any() or (x < - 1e-6).any():
        raise ValueError(f'IncreaseMultiplicityFlow operates on [0,1] but input was {x}')
