import torch
import numpy as np

# TODO: write docstrings

class SplittingLayer(torch.nn.Module):
    
    def __init__(self, n_left=1):
        super().__init__()
        self._n_left = n_left
    
    def forward(self, *xs, inverse=False):
        if len(xs) > 2:
            raise ValueError()
        if not inverse:
            x = xs[0]
            dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
            x_left = x[..., :self._n_left]
            x_right = x[..., self._n_left:]
            return x_left, x_right, dlogp
        else:
            x_left, x_right = xs
            dlogp = torch.zeros(*xs[0].shape[:-1], 1).to(xs[0])
            x = torch.cat([x_left, x_right], dim=-1)
            return x, dlogp
        

class SwappingLayer(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *xs, inverse=False):
        dlogp = torch.zeros(*xs[0].shape[:-1], 1).to(xs[0])
        if not inverse:
            xs = (*xs[1:], xs[0])
        else:
            xs = (xs[-1], *xs[0:-1])
        return (*xs, dlogp)


class InvertedLayer(torch.nn.Module):
    
    def __init__(self, delegate):
        super().__init__()
        self._delegate = delegate
    
    def forward(self, *xs, inverse=False):
        return self._delegate(*xs, inverse=(not inverse))