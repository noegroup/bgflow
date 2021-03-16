import torch

from .base import Flow

#TODO: write docstrings
__all__ = ["InverseFlow"]


class InverseFlow(Flow):
    def __init__(self, delegate):
        super().__init__()
        self._delegate = delegate

    def _forward(self, *xs, **kwargs):
        return self._delegate._inverse(*xs, **kwargs)
    
    def _inverse(self, *xs, **kwargs):
        return self._delegate._forward(*xs, **kwargs)