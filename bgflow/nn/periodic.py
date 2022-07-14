
import torch
import numpy as np
__all__ = ["WrapPeriodic"]


class WrapPeriodic(torch.nn.Module):
    """Wrap network inputs around a unit sphere.

    Parameters
    ----------
    net : torch.nn.Module
        The module, whose inputs are wrapped around the unit sphere.
    left : float, optional
        Left boundary of periodic interval.
    right : float, optional
        Right boundary of periodic interval.
    indices : Union[Sequence[int], slice], optional
        Array of periodic input indices.
        Only indices covered by this index array are wrapped around the sphere.
        The default corresponds with all indices.
    """
    def __init__(self, net, left=0.0, right=1.0, indices=slice(None)):
        super().__init__()
        self.net = net
        self.left = left
        self.right = right
        self.indices = indices

    def forward(self, x):
        indices = np.arange(x.shape[-1])[self.indices]
        other_indices = np.setdiff1d(np.arange(x.shape[-1]), indices)
        y = x[..., indices]
        cos = torch.cos(2 * np.pi * (y - self.left) / (self.right - self.left))
        sin = torch.sin(2 * np.pi * (y - self.left) / (self.right - self.left))
        x = torch.cat([cos, sin, x[..., other_indices]], dim=-1)
        return self.net.forward(x)


class WrapDistances(torch.nn.Module):
    """TODO: TEST!!!"""
    def __init__(self, net, left=0.0, right=1.0, indices=slice(None)):
        super().__init__()
        self.net = net
        self.left = left
        self.right = right
        self.indices = indices

    def forward(self, x):
        indices = np.arange(x.shape[-1])[self.indices]
        other_indices = np.setdiff1d(np.arange(x.shape[-1]), indices)
        y = x[..., indices].view(x.shape[0],-1,3)
        distance_matrix = torch.cdist(y,y)
        mask = ~torch.tril(torch.ones_like(distance_matrix)).bool()
        
        distances = distance_matrix[mask].view(x.shape[0], -1)
        x = torch.cat([x[..., other_indices], distances], dim=-1)
        return self.net.forward(x)
