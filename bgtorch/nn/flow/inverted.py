
from .base import Flow

__all__ = ["InverseFlow"]


class InverseFlow(Flow):
    """The inverse of a given transform.

    Parameters
    ----------
    delegate : Flow
        The flow to invert.
    """
    def __init__(self, delegate):
        super().__init__()
        self._delegate = delegate

    def _forward(self, *xs, **kwargs):
        return self._delegate._inverse(*xs, **kwargs)
    
    def _inverse(self, *xs, **kwargs):
        return self._delegate._forward(*xs, **kwargs)