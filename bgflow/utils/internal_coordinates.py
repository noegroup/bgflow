import torch
from torch.nn import Module
import numpy as np

def dist_np(x1, x2):
    d = x2-x1
    d2 = np.sum(d*d, axis=2)
    return np.sqrt(d2)

def dist(x1, x2):
    d = x2-x1
    d2 = torch.sum(d*d, dim=2)
    return d2.sqrt()

def angle_np(x1, x2, x3, cossin=False):
    ba = x1 - x2
    ba /= np.linalg.norm(ba, axis=2, keepdims=True)
    bc = x3 - x2
    bc /= np.linalg.norm(bc, axis=2, keepdims=True)
    cosine_angle = np.sum(ba*bc, axis=2)
    a = np.arccos(cosine_angle)
    if cossin:
        sine_angle = np.sin(a)
        return np.concatenate([cosine_angle, sine_angle], axis=-1)
    else:
        return a

def angle(x1, x2, x3, cossin=False):
    ba = x1 - x2
    ba_normalized = ba / torch.norm(ba, dim=2, keepdim=True)
    bc = x3 - x2
    bc_normalized = bc / torch.norm(bc, dim=2, keepdim=True)

    cos_angle = torch.sum(ba_normalized*bc_normalized, dim=2)
    #angle = np.float32(180.0 / np.pi) * torch.acos(cosine_angle)
    a = torch.acos(cos_angle)
    if cossin:
        sin_angle = torch.sin(a)
        return torch.cat([cos_angle, sin_angle], dim=-1)
    else:
        return a

def torsion_np(x1, x2, x3, x4, cossin=False):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=2, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0*b1, axis=2, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, axis=2, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.sum(v*w, axis=2)
    b1xv = np.cross(b1, v, axisa=2, axisb=2)
    y = np.sum(b1xv*w, axis=2)
    a = np.arctan2(y, x)
    #return np.degrees(np.arctan2(y, x))
    if cossin:
        cos_angle = torch.cos(a)
        sin_angle = torch.sin(a)
        return torch.cat([cos_angle, sin_angle], dim=-1)
    else:
        return a


def torsion(x1, x2, x3, x4, cossin=False):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1_normalized = b1 / torch.norm(b1, dim=2, keepdim=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1_normalized, dim=2, keepdim=True) * b1_normalized
    w = b2 - torch.sum(b2*b1_normalized, dim=2, keepdim=True) * b1_normalized

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*w, dim=2)
    b1xv = torch.cross(b1_normalized, v)
    y = torch.sum(b1xv*w, dim=2)
    a = torch.atan2(y, x)
    if cossin:
        cos_angle = torch.cos(a)
        sin_angle = torch.sin(a)
        return torch.cat([cos_angle, sin_angle], dim=-1)
    else:
        return a


def torsioncut_minvar(torsion):
    cuts = np.linspace(-np.pi, np.pi, 37)[:-1]
    stds = []
    for cut in cuts:
        torsion_cut = np.where(torsion < cut, torsion+2*np.pi, torsion)
        stds.append(np.std(torsion_cut))
        print(cut, stds[-1])
    stds = np.array(stds)
    stdmin = stds.min()
    minindices = np.where(stds == stdmin)[0]
    return cuts[minindices[minindices.shape[0]//2]]


def torsioncut_mindensity(torsion):
    torsion_hist, torsion_edges = np.histogram(torsion, bins=36, range=[-np.pi, np.pi])
    torsion_vals = 0.5 * (torsion_edges[:-1] + torsion_edges[1:])
    mincut = torsion_vals[torsion_hist.argmin()]
    return mincut


class Coordinates(Module):
    def __init__(self, ndim=3):
        """ Transforms flat batch into particle coordinates

        Parameters
        ----------
        ndim : int
            number of space dimensions (default 3)
        """
        super().__init__()
        self.ndim = ndim

    def forward(self, x):
        return torch.reshape(x, (x.shape[0], -1, self.ndim))


class Distances(Module):
    def __init__(self, indexes):
        """ Transforms particle coordinates into distances

        Parameters
        ----------
        indexes : list of tuples or ndarray(N, 2)
            Particle pairs for which distances should be computed

        """
        super().__init__()
        self.indexes = np.array(indexes)
        assert self.indexes.ndim == 2 and self.indexes.shape[1] == 2

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Particle configuration tensor of shape (batchsize, nparticles, dim)
        """
        return dist(x[:, self.indexes[:, 0]], x[:, self.indexes[:, 1]])


class Angles(Module):
    def __init__(self, indexes, cossin=False):
        """ Transforms particle coordinates into angles

        Parameters
        ----------
        indexes : list of triples or ndarray(N, 3)
            Particle triples for which angles should be computed
        cossin : boolean
            If true, will compute cos and sin for every angle, doubling the output size.
            If false, just the angle values will be computed

        """
        super().__init__()
        self.indexes = np.array(indexes)
        assert self.indexes.ndim == 2 and self.indexes.shape[1] == 3
        self.cossin = cossin

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Particle configuration tensor of shape (batchsize, nparticles, dim)
        """
        return angle(x[:, self.indexes[:, 0]], x[:, self.indexes[:, 1]], x[:, self.indexes[:, 2]],
                     cossin=self.cossin)


class Torsions(Module):
    def __init__(self, indexes, cossin=False):
        """ Transforms particle coordinates into torsions

        Parameters
        ----------
        indexes : list of quadruples or ndarray(N, 4)
            Particle quadruples for which torsions should be computed
        cossin : boolean
            If true, will compute cos and sin for every angle, doubling the output size.
            If false, just the angle values will be computed

        """
        super().__init__()
        self.indexes = np.array(indexes)
        assert self.indexes.ndim == 2 and self.indexes.shape[1] == 4
        self.cossin = cossin

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Particle configuration tensor of shape (batchsize, nparticles, dim)
        """
        return torsion(x[:, self.indexes[:, 0]], x[:, self.indexes[:, 1]],
                       x[:, self.indexes[:, 2]], x[:, self.indexes[:, 3]], cossin=self.cossin)


class InternalCoordinates(Module):
    def __init__(self, idx_dist=None, idx_angle=None, idx_torsion=None, cossin=False):
        super().__init__()
        if idx_dist is None and idx_angle is None and idx_torsion is None:
            raise ValueError('Need to specify either distances, angles or torsions')
        self.ics = []
        self.n_ic = 0
        if idx_dist is not None:
            self.ics.append(Distances(idx_dist))
            self.n_ic += np.shape(idx_dist)[0]
        if idx_angle is not None:
            self.ics.append(Angles(idx_angle, cossin=cossin))
            if cossin:
                self.n_ic += 2*np.shape(idx_angle)[0]
            else:
                self.n_ic += np.shape(idx_angle)[0]
        if idx_torsion is not None:
            self.ics.append(Torsions(idx_torsion, cossin=cossin))
            if cossin:
                self.n_ic += 2*np.shape(idx_torsion)[0]
            else:
                self.n_ic += np.shape(idx_torsion)[0]

    def forward(self, x):
        ics = [ic(x) for ic in self.ics]
        return torch.cat(ics, dim=-1)
