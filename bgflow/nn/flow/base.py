import torch


__all__ = ["Flow"]


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()

    def forward(self, *xs, inverse=False, **kwargs):
        """
        Forward method of the flow.
        Computes the forward or inverse direction of the flow.

        Parameters
        ----------
        xs : torch.Tensor
            Input of the flow

        inverse : boolean
            Whether to compute the forward or inverse
        """
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)
