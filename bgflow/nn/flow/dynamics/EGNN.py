import torch
import torch.nn as nn
from ....utils import distance_vectors, distances_from_vectors


class EQ_GNN(torch.nn.Module):
    def __init__(self, dim, n_particles, n_features, n_hidden=32, act_fn=nn.SiLU(), attention=True, coords_range=5):
        super().__init__()
        self.dim = dim
        self.n_particles = n_particles
        self.n_dimensions = dim // n_particles
        self.n_features = n_features
        self.attention = attention
        self.coords_range = coords_range
    
        self.edge_network = nn.Sequential(
            nn.Linear(2 * n_features + 2, n_hidden),
            act_fn,
            nn.Linear(n_hidden, n_features),
            act_fn)
        
        self.node_network = nn.Sequential(
            nn.Linear(2 * n_features, n_hidden),
            act_fn,
            nn.Linear(n_hidden, n_features),
        )
        l_layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(l_layer.weight, gain=0.001)

        self.coord_network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            act_fn,
            l_layer,
            nn.Tanh()
        )
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(n_features, 1),
                nn.Sigmoid())
            
        self.edge_idxs = self._create_edges_indices()
        
    def _create_edges_indices(self):
        """Create array of indices such that it matches the distances."""
        idxs = []
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    idxs.append(i)
                    idxs.append(j)
        return torch.LongTensor(idxs)


    def edge_transformation(self, h, d, d_static):
        #print("edge", d, d_static)
        # TODO change the reshaping, this is not neccessary
        hs = h[:, self.edge_idxs].view(-1, 2 * self.n_features)
        d = d.view(-1,1)
        d_static = d_static.view(-1,1)
        hd = torch.cat([hs, d, d_static], dim=-1)
        # [b*n *(n - 1), n_features]
        m = self.edge_network(hd)
        # self attention
        if self.attention:
            att_m = self.att_mlp(m)
            m = m * att_m
        return m
    
    def node_transformation(self, h, m_ij):
        m_i = m_ij.view(-1, self.n_particles, self.n_particles - 1, self.n_features).sum(dim=2)
        hm_i = torch.cat([h, m_i], dim=-1)
        h_update = self.node_network(hm_i)
        return h_update

    def forward(self, x, h, d_static):
        x = x.view(-1, self.n_particles, self.n_dimensions)
        r = distance_vectors(x)
        d = distances_from_vectors(r).view(-1, 1)
        m = self.edge_transformation(h, d.pow(2), d_static.pow(2))
        x_update = (r / (d.view(-1, self.n_particles, self.n_particles - 1, 1) + 1) 
                    * self.coord_network(m).view(-1, self.n_particles, self.n_particles - 1, 1)
                   )
        x = x + x_update.sum(dim=2) * self.coords_range

        h_update = self.node_transformation(h, m)
        h = h + h_update 
        #print("h", h)
        return x, h
    
class EGNN(torch.nn.Module):
    def __init__(self, dim, n_particles, n_features, n_layers=3, n_hidden=32, act_fn=nn.SiLU(), condition_time=False, coords_range=15):
        super().__init__()
        self.dim = dim
        self.n_particles = n_particles
        self.n_dimensions = dim // n_particles
        self.n_features = n_features
        self.condition_time = condition_time
        self.n_layers = n_layers
        self.coords_range = coords_range / n_layers
        self.node_embedding_network = nn.Linear(1, n_features)
        self.embedding_out = nn.Linear(n_features, 1)

        for i in range(0, n_layers):
            self.add_module("eq_gnn_%d" % i, EQ_GNN(
                dim, n_particles, n_features, n_hidden, act_fn, coords_range=self.coords_range))
                            
        # self.node_out_network = nn.Linear(n_features, n_features)
                            
    def forward(self, x):
        h = torch.ones(x.shape[0], self.n_particles, 1).to(x)
        if self.condition_time:
            h = h*t                    
        h = self.node_embedding_network(h)
        x_update = x.clone()
        x = x.view(-1, self.n_particles, self.n_dimensions)
        r = distance_vectors(x)
        d_static = distances_from_vectors(r).view(-1, 1)
        for i in range(0, self.n_layers):
            x_update, h = self._modules["eq_gnn_%d" % i](x_update, h, d_static)
        update = (x_update - x).view(-1, self.n_particles, self.n_dimensions) 
        return (update - update.mean(dim=1, keepdim=True)).view(-1, self.dim)
