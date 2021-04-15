from ..base import Flow


__all__ = ["Transformer"]


class Transformer(Flow):
    
    def __init__(self):
        super().__init__()
    
    def _forward(self, x, y, *args, **kwargs):
        raise NotImplementedError()
    
    def _inverse(self, x, y, *args, **kwargs):
        raise NotImplementedError()
