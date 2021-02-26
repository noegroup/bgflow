import torch


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()

    def forward(self, *xs, inverse=False, **kwargs):
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)
