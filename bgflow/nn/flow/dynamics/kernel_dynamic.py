import torch
import numpy as np
from ....utils import distance_vectors, distances_from_vectors, rbf_kernels


class KernelDynamics(torch.nn.Module):
    """
    Equivariant dynamics functions.
    Equivariant dynamics functions that allows an efficient
    and exact divergence computation :footcite:`Khler2020EquivariantFE`.

    References
    ----------
    .. footbibliography::

    """

    def __init__(self, n_particles, n_dimensions,
                 mus, gammas,
                 mus_time=None, gammas_time=None,
                 optimize_d_gammas=False,
                 optimize_t_gammas=False):
        super().__init__()
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions

        self._mus = mus
        self._neg_log_gammas = -torch.log(gammas)

        self._n_kernels = self._mus.shape[0]

        self._mus_time = mus_time
        self._neg_log_gammas_time = -torch.log(gammas_time)

        if self._mus_time is None:
            self._n_out = 1
        else:
            assert self._neg_log_gammas_time is not None and self._neg_log_gammas_time.shape[0] == self._mus_time.shape[
                0]
            self._n_out = self._mus_time.shape[0]

        if optimize_d_gammas:
            self._neg_log_gammas = torch.nn.Parameter(self._neg_log_gammas)

        if optimize_t_gammas:
            self._neg_log_gammas_time = torch.nn.Parameter(self._neg_log_gammas_time)

        self._weights = torch.nn.Parameter(
            torch.Tensor(self._n_kernels, self._n_out).normal_() * np.sqrt(1. / self._n_kernels)
        )
        self._bias = torch.nn.Parameter(
            torch.Tensor(1, self._n_out).zero_()
        )

        self._importance = torch.nn.Parameter(
            torch.Tensor(self._n_kernels).zero_()
        )

    def _force_mag(self, t, d, derivative=False):

        importance = self._importance

        rbfs, d_rbfs = rbf_kernels(d, self._mus, self._neg_log_gammas, derivative=derivative)

        force_mag = (rbfs + importance.pow(2).view(1, 1, 1, -1)) @ self._weights + self._bias
        if derivative:
            d_force_mag = (d_rbfs) @ self._weights
        else:
            d_force_mag = None
        if self._mus_time is not None:
            trbfs, _ = rbf_kernels(t, self._mus_time, self._neg_log_gammas_time)
            force_mag = (force_mag * trbfs).sum(dim=-1, keepdim=True)
            if derivative:
                d_force_mag = (d_force_mag * trbfs).sum(dim=-1, keepdim=True)
        return force_mag, d_force_mag

    def forward(self, t, x, compute_divergence=True):
        """
        Computes the change of the system `dxs` at state `x` and
        time `t` due to the kernel dynamic. Furthermore, can also compute the exact change of log density
        which is equal to the divergence of the change.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        x : PyTorch tensor
            The current configuration of the system
        compute_divergence : boolean
            Whether the divergence is computed

        Returns
        -------
        forces, -divergence: PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`forces`) and the negative exact update of the log density (`-divergence`)

        """
        n_batch = x.shape[0]

        x = x.view(n_batch, self._n_particles, self._n_dimensions)
        r = distance_vectors(x)

        d = distances_from_vectors(r).unsqueeze(-1)

        force_mag, d_force_mag = self._force_mag(t, d, derivative=compute_divergence)
        forces = (r * force_mag).sum(dim=-2)
        forces = forces.view(n_batch, -1)

        if compute_divergence:
            divergence = (d * d_force_mag + self._n_dimensions * force_mag).view(n_batch, -1).sum(dim=-1)
            divergence = divergence.unsqueeze(-1)
            return forces, -divergence
        else:
            return forces
