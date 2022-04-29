
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

class WrapDistancesIC(torch.nn.Module):
    """TODO: TEST!!!"""
    def __init__(self, net, coordinate_transform, cdf_flow, compute_transform, original_conditioner, left=0.0, right=1.0, indices=slice(None)):
        super().__init__()
        self.net = net
        self.left = left
        self.right = right
        self.indices = indices
        self.coordinate_transform = coordinate_transform
        self.cdf_flow = cdf_flow
        self.compute_transform = compute_transform 
        self.original_conditioner = original_conditioner
        self.dim_bonds = coordinate_transform.dim_bonds
        self.dim_angles = coordinate_transform.dim_angles
        self.dim_torsions = coordinate_transform.dim_torsions
        self.distances = None
        import warnings
        warnings.warn("WrapDistances is not tested!")

    def forward(self, x):

        #indices = np.arange(x.shape[-1])[self.indices]
        indices = self.indices
        other_indices = np.setdiff1d(np.arange(x.shape[-1]), indices)
        
       # import ipdb
       # ipdb.set_trace()        

        if self.compute_transform:
            ics = x[..., indices]
            bonds_01 = ics[:,:self.dim_bonds]
            angles_01 = ics[:,self.dim_bonds:self.dim_angles+self.dim_angles+1]
            torsions_01 = ics[:,-self.dim_torsions:]
            x0 = torch.zeros(x.shape[0], 3).to(bonds_01)
            R = torch.ones(x.shape[0], 3).to(bonds_01)*0.5        
            bonds = self.cdf_flow[0](bonds_01)[0]
            angles = self.cdf_flow[1](angles_01)[0]
            torsions = self.cdf_flow[2](torsions_01)[0]
            cart = self.coordinate_transform(*(bonds, angles,torsions, x0, R), inverse = True)[0]
            cart = cart.view(cart.shape[0],-1,3)
            distance_matrix = torch.cdist(cart,cart)
            distances = distance_matrix.view(x.shape[0], -1)
            self.distances = distances
 
        else:
            #import ipdb
            #ipdb.set_trace() 
            distances = self.original_conditioner.transformer._params_net.net.distances
        

        #import ipdb
        #ipdb.set_trace() 
        x = torch.cat([x, distances], dim=-1)
        return self.net.forward(x)


class WrapDistances(torch.nn.Module):
    """TODO: TEST!!!"""
    def __init__(self, net, left=0.0, right=1.0, indices=slice(None)):
        super().__init__()
        self.net = net
        self.left = left
        self.right = right
        self.indices = indices
        import warnings
        warnings.warn("WrapDistances is not tested!")

    def forward(self, x):
        #import ipdb
        #ipdb.set_trace()
        indices = np.arange(x.shape[-1])[self.indices]
        other_indices = np.setdiff1d(np.arange(x.shape[-1]), indices)
        y = x[..., indices].view(x.shape[0],-1,3)
        distance_matrix = torch.cdist(y,y)
        mask = ~torch.tril(torch.ones_like(distance_matrix)).bool()
        
        distances = distance_matrix[mask].view(x.shape[0], -1)
        #import pytest
        #pytest.set_trace()
        x = torch.cat([x[..., other_indices], distances], dim=-1)
        #import pytest
        #pytest.set_trace()
        return self.net.forward(x)
