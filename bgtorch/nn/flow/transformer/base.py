from ..base import Flow

class Transformer(Flow):
    
    def __init__(self):
        super().__init__()
    
    def _forward(self, y, *args, **kwargs):
        raise NotImplementedError()
    
    def _inverse(self, y, *args, **kwargs):
        raise NotImplementedError()
    