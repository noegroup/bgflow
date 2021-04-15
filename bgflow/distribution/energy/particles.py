import numpy as np
import torch

from .base import Energy

__all__ = ["RepulsiveParticles", "HarmonicParticles"]


MAX_BATCHSIZE_NUMPY = 10000
MAX_BATCHSIZE_TORCH = 100000


def ensure_traj(X):
    if np.ndim(X) == 2:
        return X
    if np.ndim(X) == 1:
        return np.array([X])
    raise ValueError("Incompatible array with shape: ", np.shape(X))


def distance_matrix_squared(crd1, crd2, dim=2):
    """ Returns the distance matrix or matrices between particles
    Parameters
    ----------
    crd1 : array or matrix
        first coordinate set
    crd2 : array or matrix
        second coordinate set
    dim : int
        dimension of particle system. If d=2, coordinate vectors are
        [x1, y1, x2, y2, ...]
    """
    crd1 = ensure_traj(crd1)
    crd2 = ensure_traj(crd2)
    n = int(np.shape(crd1)[1] / dim)

    crd1_components = [
        np.tile(np.expand_dims(crd1[:, i::dim], 2), (1, 1, n)) for i in range(dim)
    ]
    crd2_components = [
        np.tile(np.expand_dims(crd2[:, i::dim], 2), (1, 1, n)) for i in range(dim)
    ]
    D2_components = [
        (crd1_components[i] - np.transpose(crd2_components[i], axes=(0, 2, 1))) ** 2
        for i in range(dim)
    ]
    D2 = np.sum(D2_components, axis=0)
    return D2


class RepulsiveParticles(Energy):
    params_default = {
        "nsolvent": 36,
        "eps": 1.0,  # LJ prefactor
        "rm": 1.1,  # LJ particle size
        "dimer_slope": -1,  # dimer slope parameter
        "dimer_a": 25.0,  # dimer x2 parameter
        "dimer_b": 10.0,  # dimer x4 parameter
        "dimer_dmid": 1.5,  # dimer transition state distance
        "dimer_k": 20.0,  # dimer force constant
        "box_halfsize": 3.0,
        "box_k": 100.0,  # box repulsion force constant
        "grid_k": 0.0,  # restraint strength to particle grid (to avoid permutation)
        "rc": 0.9,  # cutoff for the surrogate energy
    }

    def __init__(self, params=None):
        self.nparticles = params["nsolvent"] + 2
        dim = 2 * self.nparticles
        # set parameters
        super().__init__(dim)
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables

        self.rm = self.params["rm"]
        self.rm12 = self.params["rm"] ** 12

        self.a_surrogate = 21.0 * self.params["rm"] ** 6 / self.params["rc"] ** 8
        self.b_surrogate = 6.0 * self.params["rm"] ** 6 / self.params["rc"] ** 7
        self.c_surrogate = self.params["rm"] ** 6 / self.params["rc"] ** 6

        # create mask matrix to help computing particle interactions
        self.mask_matrix = np.ones((self.nparticles, self.nparticles), dtype=np.float32)
        self.mask_matrix[0, 1] = 0.0
        self.mask_matrix[1, 0] = 0.0
        for i in range(self.nparticles):
            self.mask_matrix[i, i] = 0.0
        self.mask_matrix_torch = torch.from_numpy(self.mask_matrix)

    def dimer_distance(self, x):
        return np.sqrt((x[:, 2] - x[:, 0]) ** 2 + (x[:, 3] - x[:, 1]) ** 2)

    def _distance_squared_matrix(self, crd1, crd2):
        return distance_matrix_squared(crd1, crd2, dim=2)

    def LJ_energy_torch(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = x.shape[0]
        n = xcomp.shape[1]
        Xcomp = xcomp.unsqueeze(1).repeat([1, n, 1])
        Ycomp = ycomp.unsqueeze(1).repeat([1, n, 1])
        Dx = Xcomp - torch.transpose(Xcomp, 1, 2)
        Dy = Ycomp - torch.transpose(Ycomp, 1, 2)
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = torch.Tensor.repeat(
            torch.unsqueeze(self.mask_matrix_torch.to(D2), 0), [batchsize, 1, 1]
        )
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.params["rm"] ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        # energy
        E = (
            0.5 * self.params["eps"] * torch.sum(D2rel ** 6, dim=(1, 2))
        )  # do 1/2 because we have double-counted each interaction
        return E

    def LJ_energy_surrogate_torch(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = x.shape[0]
        n = xcomp.shape[1]
        Xcomp = xcomp.unsqueeze(1).repeat([1, n, 1])
        Ycomp = ycomp.unsqueeze(1).repeat([1, n, 1])
        Dx = Xcomp - torch.transpose(Xcomp, 1, 2)
        Dy = Ycomp - torch.transpose(Ycomp, 1, 2)
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = torch.Tensor.repeat(
            torch.unsqueeze(self.mask_matrix_torch.to(D2), 0), [batchsize, 1, 1]
        )
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = (self.params["rm"] ** 2) / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        distance_mask = D2 > self.params["rc"] ** 2
        distance_mask = distance_mask.to(D2)
        D = torch.sqrt(D2)
        E_h = (
            self.a_surrogate * (D - self.params["rc"]) ** 2
            - self.b_surrogate * (D - self.params["rc"])
            + self.c_surrogate
        )
        E_h *= 1 - distance_mask
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix * distance_mask
        # energy
        # do 1/2 because we have double-counted each interaction
        E_LJ = 0.5 * self.params["eps"] * torch.sum(D2rel ** 6, dim=(1, 2))
        return E_LJ + 0.5 * torch.sum(E_h, dim=(1, 2))

    def LJ_force_torch(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = x.shape[0]
        n = xcomp.shape[1]
        Xcomp = xcomp.unsqueeze(1).repeat([1, n, 1])
        Ycomp = ycomp.unsqueeze(1).repeat([1, n, 1])
        Dx = Xcomp - torch.transpose(Xcomp, 1, 2)
        Dy = Ycomp - torch.transpose(Ycomp, 1, 2)
        D2 = Dx ** 2 + Dy ** 2
        mmatrix = torch.Tensor.repeat(
            torch.unsqueeze(self.mask_matrix_torch.to(D2), 0), [batchsize, 1, 1]
        )
        D2 = D2 + (
            1.0 - mmatrix
        )  # this is just to avoid NaNs, the inverses will be set to 0 later
        D2rel = 1.0 / D2
        # remove self-interactions and interactions between dimer particles
        D2rel = D2rel * mmatrix
        # energy
        Fx = (
            self.params["eps"] * self.rm12 * torch.sum(D2rel ** 7 * Dx, dim=(2))
        )  # do 1/2 because we have double-counted each interaction
        Fy = (
            self.params["eps"] * self.rm12 * torch.sum(D2rel ** 7 * Dy, dim=(2))
        )  # do 1/2 because we have double-counted each interaction
        F = torch.cat([Fx.unsqueeze(2), Fy.unsqueeze(2)], dim=2)
        return -12 * F.reshape([batchsize, self.dim])

    def dimer_energy_torch(self, x):
        # center restraint energy
        energy_dx = self.params["dimer_k"] * (x[:, 0] + x[:, 2]) ** 2
        # y restraint energy
        energy_dy = (
            self.params["dimer_k"] * (x[:, 1]) ** 2
            + self.params["dimer_k"] * (x[:, 3]) ** 2
        )
        # first two particles
        d = torch.sqrt((x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2)
        d0 = 2 * (d - self.params["dimer_dmid"])
        d2 = d0 * d0
        d4 = d2 * d2
        energy_interaction = (
            self.params["dimer_slope"] * d0
            - self.params["dimer_a"] * d2
            + self.params["dimer_b"] * d4
        )

        return energy_dx + energy_dy + energy_interaction

    def dimer_force_torch(self, x):
        F = torch.zeros_like(x)
        # center restraint energy
        F[:, 0] += -self.params["dimer_k"] * (x[:, 0] + x[:, 2]) * 2
        F[:, 2] += -self.params["dimer_k"] * (x[:, 0] + x[:, 2]) * 2
        # y restraint energy
        F[:, 1] += -self.params["dimer_k"] * (x[:, 1]) * 2
        F[:, 3] += -self.params["dimer_k"] * (x[:, 3]) * 2
        # first two particles
        d = x[:, :2] - x[:, 2:4]
        r = torch.sqrt((d[:, 0]) ** 2 + (d[:, 1]) ** 2)
        dhat = d / r.unsqueeze(1)
        d1 = 2 * (r - self.params["dimer_dmid"])
        d3 = d1 ** 3
        F_dimer = (
            -2 * self.params["dimer_slope"]
            + 4 * self.params["dimer_a"] * d1
            - 8 * self.params["dimer_b"] * d3
        ).unsqueeze(1) * dhat
        F[:, :2] += F_dimer
        F[:, 2:4] -= F_dimer
        return F

    def box_energy_torch(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        # indicator functions
        E = 0.0
        d_left = -(xcomp + self.params["box_halfsize"])
        E += torch.sum(
            (torch.sign(d_left) + 1) * self.params["box_k"] * d_left ** 2, dim=1
        )
        d_right = xcomp - self.params["box_halfsize"]
        E += torch.sum(
            (torch.sign(d_right) + 1) * self.params["box_k"] * d_right ** 2, dim=1
        )
        d_down = -(ycomp + self.params["box_halfsize"])
        E += torch.sum(
            (torch.sign(d_down) + 1) * self.params["box_k"] * d_down ** 2, dim=1
        )
        d_up = ycomp - self.params["box_halfsize"]
        E += torch.sum((torch.sign(d_up) + 1) * self.params["box_k"] * d_up ** 2, dim=1)
        return E

    def box_force_torch(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        # indicator functions
        Fx = torch.zeros_like(xcomp)
        Fy = torch.zeros_like(xcomp)
        d_left = -(xcomp + self.params["box_halfsize"])
        Fx += 2 * (torch.sign(d_left) + 1) * self.params["box_k"] * d_left
        d_right = xcomp - self.params["box_halfsize"]
        Fx += -2 * (torch.sign(d_right) + 1) * self.params["box_k"] * d_right
        d_down = -(ycomp + self.params["box_halfsize"])
        Fy += 2 * (torch.sign(d_down) + 1) * self.params["box_k"] * d_down
        d_up = ycomp - self.params["box_halfsize"]
        Fy += -2 * (torch.sign(d_up) + 1) * self.params["box_k"] * d_up
        return torch.cat([Fx.unsqueeze(2), Fy.unsqueeze(2)], dim=2).reshape((-1, 76))

    def _energy(self, x):
        return (
            self.LJ_energy_torch(x)
            + self.dimer_energy_torch(x)
            + self.box_energy_torch(x)
        ).view(-1, 1)

    def _surrogate_energy(self, x):
        return (
            self.LJ_energy_surrogate_torch(x)
            + self.box_energy_torch(x)
            + self.dimer_energy_torch(x)
        )

    def surrogate_energy(self, x):
        if x.shape[0] < MAX_BATCHSIZE_TORCH:
            return self._surrogate_energy(x)
        energy_x = torch.zeros(x.shape[0]).to(x)
        for i in range(0, len(energy_x), MAX_BATCHSIZE_TORCH):
            i_from = i
            i_to = min(i_from + MAX_BATCHSIZE_TORCH, len(energy_x))
            energy_x[i_from:i_to] = self._surrogate_energy(x[i_from:i_to])
        return energy_x

    def plot_dimer_energy(self, axis=None):
        """ Plots the dimer energy to the standard figure """
        x_scan = np.linspace(0.5, 2.5, 100)
        E_scan = self.dimer_energy(
            np.array([-0.5 * x_scan, np.zeros(100), 0.5 * x_scan, np.zeros(100)]).T
        )
        E_scan -= E_scan.min()

        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()
        # plt.figure(figsize=(5, 4))
        axis.plot(x_scan, E_scan, linewidth=2)
        axis.set_xlabel("x / a.u.")
        axis.set_ylabel("Energy / kT")
        axis.set_ylim(E_scan.min() - 2.0, E_scan[int(E_scan.size / 2)] + 2.0)

        return x_scan, E_scan

    def forward(self, t, x):
        q = x[0].view(-1, self.dim)
        q.requires_grad = True
        p = x[1].view(-1, self.dim)
        E = self.energy_torch(q.view(-1, self.dim))
        E.backward()
        return torch.cat([p, -q.grad], dim=0)

    def force(self, x):
        return (
            self.LJ_force_torch(x) + self.dimer_force_torch(x) + self.box_force_torch(x)
        )

    def force_autograd(self, x):
        E = self.energy_torch(x)
        ones = torch.ones(x.shape[0]).to(x)
        return -torch.autograd.grad(
            E, x, ones, retain_graph=True, create_graph=True, allow_unused=True
        )[0]

    def hamiltonian(self, mu):
        x = mu[:, : self.dim]
        p = mu[:, self.dim : 2 * self.dim]
        return self.energy_torch(x) + torch.sum(p ** 2, dim=1) / 2.0

    def surrogate_hamiltonian(self, mu):
        x = mu[:, : self.dim]
        p = mu[:, self.dim : 2 * self.dim]
        return self.surrogate_energy_torch(x) + torch.sum(p ** 2, dim=1) / 2.0


class HarmonicParticles(RepulsiveParticles):
    def __init__(self, spring_constant=200.0, params=None):
        if params is None:
            params = RepulsiveParticles.params_default
        super().__init__(params)
        self.spring_constant = spring_constant

    def harmonic_energy_torch(self, x):
        # all component-wise distances bet
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = x.shape[0]
        n = xcomp.shape[1]
        Xcomp = xcomp.unsqueeze(1).repeat([1, n, 1])
        Ycomp = ycomp.unsqueeze(1).repeat([1, n, 1])
        Dx = Xcomp - torch.transpose(Xcomp, 1, 2)
        Dy = Ycomp - torch.transpose(Ycomp, 1, 2)
        D2 = Dx ** 2 + Dy ** 2
        distance_mask = D2 < self.params["rc"] ** 2
        distance_mask = distance_mask.to(D2)
        mmatrix = torch.Tensor.repeat(
            torch.unsqueeze(self.mask_matrix_torch.to(D2), 0), [batchsize, 1, 1]
        )
        D = torch.sqrt(D2)
        E = self.spring_constant * (D - self.params["rc"]) ** 2
        E *= distance_mask * mmatrix
        # remove self-interactions and interactions between dimer particles
        return 0.5 * torch.sum(E, dim=(1, 2))

    def _energy(self, x):
        return (
            self.harmonic_energy_torch(x)
            + self.dimer_energy_torch(x)
            + self.box_energy_torch(x)
        ).view(-1, 1)
