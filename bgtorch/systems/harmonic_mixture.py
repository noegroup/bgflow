import torch
import numpy as np

from .base import EnergyBasedModel


# TODO: write docstrings


class MixtureModel(EnergyBasedModel):
    
    def __init__(self, n_dims, n_clusters=5, spread=1, temperature=1., excentricity=0.4):
        super().__init__()
        self._temperature = temperature
        self._n_clusters = n_clusters
        self._n_dims = n_dims
        self._mus = torch.Tensor(n_clusters, n_dims).normal_() * spread
        cov = np.random.normal(size=(n_clusters, n_dims, n_dims))
        cov = np.einsum("nij, nkj -> nik", cov, cov)
        d, u = np.linalg.eig(cov)
        self._d = torch.Tensor(n_clusters, n_dims).uniform_(1. - excentricity, 1. + excentricity)
        self._u = torch.Tensor(u)
        self._weights = torch.softmax(torch.Tensor(n_clusters).normal_(), dim=0)
        
    def sample(self, sample_shape, temperature=None):
        if temperature is None:
            temperature = self._temperature
        i = np.random.choice(self._n_clusters, sample_shape, p=self._weights.numpy()).reshape(-1)
        i = torch.Tensor(i).long()
        mu = self._mus[i].view(-1, self._n_dims)
        u = self._u[i].view(-1, self._n_dims, self._n_dims)
        d = self._d[i].view(-1, self._n_dims)
        eps = torch.Tensor(*sample_shape, self._n_dims).normal_().view(-1, self._n_dims)
        eps = torch.einsum("nij, nj, nj -> ni", u, eps, d)
        z = mu + eps * np.sqrt(temperature)
        return z.view(*sample_shape, self._n_dims) 
    
    def energy(self, x, temperature=None): 
        if temperature is None:
            temperature = self._temperature
        d_inv = 1./self._d
        d_inv = d_inv.unsqueeze(0)
        ut = self._u.permute(0, 2, 1)
        mus = self._mus.unsqueeze(0)
        x = x.unsqueeze(1)
        eps = d_inv * (x - mus)
        eps = torch.einsum("bnd, nde -> bne", eps, self._u)
        energy = -0.5 * torch.einsum("bnd, bnd -> bn", eps, eps) / temperature
        energy = -torch.logsumexp(energy + torch.log(self._weights).unsqueeze(0), dim=-1)
        energy = energy.view(-1, 1)
        return energy
        