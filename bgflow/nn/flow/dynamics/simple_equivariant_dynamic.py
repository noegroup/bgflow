import torch
from ....utils import distance_vectors, distances_from_vectors


class InvariantNet(torch.nn.Module):    
    def __init__(self, n_particles, n_dof, dist_net, encoder=None):
        super().__init__()
        self._dist_net = dist_net
        self._encoder = encoder
        self._n_particles = n_particles
        self._n_dof = n_dof    
    def forward(self, x):
        n_batch = x.shape[0]
        n_dim = self._n_particles * self._n_dof
        assert x.shape[-1] == n_dim
        x = x.view(n_batch, self._n_particles, self._n_dof)
        r = distance_vectors(x) 
        d = distances_from_vectors(r)
        if self._encoder is not None:
            d = self._encoder(d.unsqueeze(-1))
        d_shape = d.shape
        f = self._dist_net(d.view(-1, d_shape[-1]))
        f = f.view(*d_shape[:-1], -1)
        f = f.view(n_batch, -1).mean(dim=-1, keepdim=True) * torch.ones(n_batch, n_dim).cuda()
        return f
    
class EquivariantNet(torch.nn.Module):          
    def __init__(self, n_particles, n_dof, dist_net, encoder=None, remove_mean=True):
        super().__init__()
        self._dist_net = dist_net
        self._encoder = encoder
        self._invariant_net = dist_net
        self._n_particles = n_particles
        self._n_dof = n_dof
        self._remove_mean = remove_mean  
        
    def forward(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dof)
        r = distance_vectors(x) 
        dist = distances_from_vectors(r)
        r = r / (dist.unsqueeze(-1) + 1e-3)
        if self._encoder is not None:
            d = self._encoder(dist.unsqueeze(-1)) 
        d_shape = d.shape
        f = self._dist_net(d.view(-1, d_shape[-1]))
        f = f.view(*d_shape[:-1], -1) 
        f = (f * r).sum(dim=-2)
        if self._remove_mean:
            f = remove_mean(f, self._n_particles, self._n_dof)
        return f.view(n_batch, -1)
