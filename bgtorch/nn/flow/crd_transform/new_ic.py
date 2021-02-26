import torch
import numpy as np

from ..base import Flow
from .ic_helper import (
    outer,
    dist_deriv,
    angle_deriv,
    torsion_deriv,
    orientation,
    det3x3,
    det2x2,
)
from .pca import WhitenFlow


def ic2xyz_deriv(p1, p2, p3, d14, a124, t1234):
    """ computes the xyz coordinates from internal coordinates
        relative to points `p1`, `p2`, `p3` together with its
        jacobian with respect to `p1`.
    """

    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=-1)
    nn = torch.cross(v1, n, dim=-1)

    n_normalized = n / torch.norm(n, dim=-1, keepdim=True)
    nn_normalized = nn / torch.norm(nn, dim=-1, keepdim=True)

    n_scaled = n_normalized * -torch.sin(t1234)
    nn_scaled = nn_normalized * torch.cos(t1234)

    v3 = n_scaled + nn_scaled
    v3_norm = torch.norm(v3, dim=-1, keepdim=True)
    v3_normalized = v3 / v3_norm
    v3_scaled = v3_normalized * d14 * torch.sin(a124)

    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    v1_normalized = v1 / v1_norm
    v1_scaled = v1_normalized * d14 * torch.cos(a124)

    position = p1 + v3_scaled - v1_scaled

    J_d = v3_normalized * torch.sin(a124) - v1_normalized * torch.cos(a124)
    J_a = v3_normalized * d14 * torch.cos(a124) + v1_normalized * d14 * torch.sin(a124)

    J_t1 = (d14 * torch.sin(a124))[..., None]
    J_t2 = (
        1.0
        / v3_norm[..., None]
        * (torch.eye(3)[None, :].to(p1) - outer(v3_normalized, v3_normalized))
    )

    J_n_scaled = n_normalized * -torch.cos(t1234)
    J_nn_scaled = nn_normalized * -torch.sin(t1234)
    J_t3 = (J_n_scaled + J_nn_scaled)[..., None]

    J_t = (J_t1 * J_t2) @ J_t3

    J = torch.stack([J_d, J_a, J_t[..., 0]], dim=-1)

    return position, J


def ic2xy0_deriv(p1, p2, d14, a124):
    """ computes the xy coordinates (z set to 0) for the given
        internal coordinates together with the Jacobian
        with respect to `p1`.
    """

    t1234 = torch.Tensor([[0.5 * np.pi]]).to(p1)
    p3 = torch.Tensor([[0, -1, 0]]).to(p1)
    xyz, J = ic2xyz_deriv(p1, p2, p3, d14, a124, t1234)
    J = J[..., [0, 1, 2], :][..., [0, 1]]
    return xyz, J


def decompose_z_matrix(z_matrix, fixed):
    atoms = [fixed]

    blocks = []
    given = np.sort(fixed)

    # filter out conditioned variables
    non_given = ~np.isin(z_matrix[:, 0], given)
    z_matrix = z_matrix[non_given]
    z_matrix = np.concatenate([np.arange(len(z_matrix))[:, None], z_matrix], axis=1)

    order = []
    while len(z_matrix) > 0:

        is_satisfied = np.all(np.isin(z_matrix[:, 2:], given), axis=-1)
        if (not np.any(is_satisfied)) and len(z_matrix) > 0:
            raise ValueError("Not satisfiable")

        pos = z_matrix[is_satisfied, 0]
        atom = z_matrix[is_satisfied, 1]

        atoms.append(atom)
        order.append(pos)

        blocks.append(z_matrix[is_satisfied][:, 1:])
        given = np.union1d(given, atom)
        z_matrix = z_matrix[~is_satisfied]

    index2atom = np.concatenate(atoms)
    atom2index = np.argsort(index2atom)
    index2order = np.concatenate(order)
    return blocks, index2atom, atom2index, index2order


def slice_initial_atoms(z_matrix):
    s = np.sum(z_matrix == -1, axis=-1)
    order = np.argsort(s)[::-1][:3]
    return z_matrix[:, 0][order], z_matrix[s == 0]


def normalize_torsions(torsions):
    period = 2 * np.pi
    torsions = (torsions + period / 2) / period
    dlogp = -np.log(period) * (torsions.shape[-1])
    return torsions, dlogp


def normalize_angles(angles):
    period = np.pi
    angles = angles / period
    dlogp = -np.log(period) * (angles.shape[-1])
    return angles, dlogp


def unnormalize_torsions(torsions):
    period = 2 * np.pi
    torsions = torsions * (period) - period / 2
    dlogp = np.log(period) * (torsions.shape[-1])
    return torsions, dlogp


def unnormalize_angles(angles):
    period = np.pi
    angles = angles * period
    dlogp = np.log(period) * (angles.shape[-1])
    return angles, dlogp


class ReferenceSystemTranformation(Flow):
    def __init__(self, normalize_angles=True):
        super().__init__()
        self._normalize_angles = normalize_angles

    def _forward(self, x0, x1, x2):

        R = orientation(x0, x1, x2)
        d01, _ = dist_deriv(x0, x1)
        d12, _ = dist_deriv(x1, x2)
        a012, _ = angle_deriv(x0, x1, x2)

        dlogp = 0

        _, _, _, neg_dlogp = self._init_points(x0, R, d01, d12, a012)

        if self._normalize_angles:
            a012, dlogp_a = normalize_angles(a012)
            dlogp += dlogp_a

        dlogp += -neg_dlogp

        return x0, R, d01, d12, a012, dlogp

    def _init_points(self, x0, R, d01, d12, a012):
        n_batch = d01.shape[0]
        dlogp = 0

        # first point placed in origin
        p0 = torch.zeros(n_batch, 1, 3).to(d01)

        # second point placed in z-axis
        p1 = torch.zeros_like(x0).to(d01)
        p1[..., 2] = d01

        # third point placed wrt to p0 and p1
        p2, J = ic2xy0_deriv(p1, p0, d12[:, None], a012[:, None])
        dlogp += det2x2(J[..., [0, 2], :]).abs().log()

        # bring back to original reference frame
        x1 = torch.einsum("bnd, bned -> bne", p1, R) + x0
        x2 = torch.einsum("bnd, bned -> bne", p2, R) + x0

        return x0, x1, x2, dlogp

    def _inverse(self, x0, R, d01, d12, a012):
        dlogp = 0

        if self._normalize_angles:
            a012, dlogp_a = unnormalize_angles(a012)
            dlogp += dlogp_a

        *res, dlogp_b = self._init_points(x0, R, d01, d12, a012)
        dlogp += dlogp_b
        return (*res, dlogp)


class RelativeInternalCoordinatesTransformation(Flow):
    """ global internal coordinate transformation:
        transforms a system from xyz to ic coordinates and back.
    """

    def __init__(self, z_matrix, fixed_atoms, normalize_angles=True):
        super().__init__()

        self._z_matrix = z_matrix

        self._fixed_atoms = fixed_atoms

        (
            self._z_blocks,
            self._index2atom,
            self._atom2index,
            self._index2order,
        ) = decompose_z_matrix(z_matrix, fixed_atoms)

        self._bond_indices = self._z_matrix[:, :2]
        self._angle_indices = self._z_matrix[:, :3]
        self._torsion_indices = self._z_matrix[:, :4]

        self._normalize_angles = normalize_angles

    def _forward(self, x, with_pose=True, *args, **kwargs):
        """ Computes xyz -> ic

            Returns bonds, angles, torsions and fixed coordinates.
        """

        n_batch = x.shape[0]
        x = x.view(n_batch, -1, 3)

        # compute bonds, angles, torsions
        # together with jacobians (wrt. to diagonal atom)
        bonds, jbonds = dist_deriv(
            x[:, self._z_matrix[:, 0]], x[:, self._z_matrix[:, 1]]
        )
        angles, jangles = angle_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
        )
        torsions, jtorsions = torsion_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
            x[:, self._z_matrix[:, 3]],
        )

        # slice fixed coordinates needed to reconstruct the system
        x_fixed = x[:, self._fixed_atoms].view(n_batch, -1)

        # aggregated induced volume change
        dlogp = 0.0

        # transforms angles from [-pi, pi] to [0, 1]
        if self._normalize_angles:
            angles, dlogp_a = normalize_angles(angles)
            torsions, dlogp_t = normalize_torsions(torsions)
            dlogp += dlogp_a + dlogp_t

        # compute volume change
        j = torch.stack([jbonds, jangles, jtorsions], dim=-2)
        dlogp += det3x3(j).abs().log().sum(dim=1, keepdim=True)

        return bonds, angles, torsions, x_fixed, dlogp

    def _inverse(self, bonds, angles, torsions, x_fixed, **kwargs):

        # aggregated induced volume change
        dlogp = 0

        # transforms angles from [0, 1] to [-pi, pi]
        if self._normalize_angles:
            angles, dlogp_a = unnormalize_angles(angles)
            torsions, dlogp_t = unnormalize_torsions(torsions)
            dlogp += dlogp_a + dlogp_t

        n_batch = x_fixed.shape[0]

        # initial points are the fixed points
        ps = x_fixed.view(n_batch, -1, 3).clone()

        # blockwise reconstruction of points left
        for block in self._z_blocks:

            # map atoms from z matrix
            # to indices in reconstruction order
            ref = self._atom2index[block]

            # slice three context points
            # from the already reconstructed
            # points using the indices
            context = ps[:, ref[:, 1:]]
            p0 = context[:, :, 0]
            p1 = context[:, :, 1]
            p2 = context[:, :, 2]

            # obtain index of currently placed
            # point in original z-matrix
            idx = self._index2order[ref[:, 0] - len(self._fixed_atoms)]

            # get bonds, angles, torsions
            # using this z-matrix index
            b = bonds[:, idx, None]
            a = angles[:, idx, None]
            t = torsions[:, idx, None]

            # now we have three context points
            # and correct ic values to reconstruct the current point
            p, J = ic2xyz_deriv(p0, p1, p2, b, a, t)

            # compute jacobian
            dlogp += det3x3(J).abs().log().sum(-1)[:, None]

            # update list of reconstructed points
            ps = torch.cat([ps, p], dim=1)

        # finally make sure that atoms are sorted
        # from reconstruction order to original order
        ps = ps[:, self._atom2index]

        return ps.view(n_batch, -1), dlogp


class GlobalInternalCoordinateTransformation(Flow):
    def __init__(self, z_matrix, normalize_angles=True):
        super().__init__()

        # find initial atoms
        initial_atoms, z_matrix = slice_initial_atoms(z_matrix)

        self._rel_ic = RelativeInternalCoordinatesTransformation(
            z_matrix=z_matrix,
            fixed_atoms=initial_atoms,
            normalize_angles=normalize_angles,
        )
        self._ref_ic = ReferenceSystemTranformation(normalize_angles=normalize_angles)

    def _forward(self, x):
        n_batch = x.shape[0]

        x = x.view(n_batch, -1, 3)

        # transform relative system wrt reference system
        bonds, angles, torsions, x_fixed, dlogp_rel = self._rel_ic(x)

        x_fixed = x_fixed.view(n_batch, -1, 3)

        # transform reference system
        x0, R, d01, d12, a012, dlogp_ref = self._ref_ic(
            x_fixed[:, [0]], x_fixed[:, [1]], x_fixed[:, [2]]
        )

        # gather bonds and angles
        bonds = torch.cat([d01, d12, bonds], dim=-1)

        angles = torch.cat([a012, angles], dim=-1)

        # aggregate volume change
        dlogp = dlogp_rel + dlogp_ref

        return bonds, angles, torsions, x0, R, dlogp

    def _inverse(self, bonds, angles, torsions, x0, R):

        # get ics of reference system
        d01 = bonds[:, [0]]
        d12 = bonds[:, [1]]
        a012 = angles[:, [0]]

        # transform reference system back
        x0, x1, x2, dlogp_ref = self._ref_ic(x0, R, d01, d12, a012, inverse=True)
        x_init = torch.cat([x0, x1, x2], dim=1)

        # now transform relative system wrt reference system back
        x, dlogp_rel = self._rel_ic(
            bonds[:, 2:], angles[:, 1:], torsions, x_init, inverse=True
        )

        # aggregate volume change
        dlogp = dlogp_rel + dlogp_ref

        return x, dlogp


class MixedCoordinateTransformation(Flow):
    def __init__(
        self, data, z_matrix, fixed_atoms, keepdims=None, normalize_angles=True
    ):
        super().__init__()
        self._whiten = self._setup_whitening_layer(data, fixed_atoms, keepdims=keepdims)
        self._rel_ic = RelativeInternalCoordinatesTransformation(
            z_matrix, fixed_atoms, normalize_angles
        )

    def _setup_whitening_layer(self, data, fixed_atoms, keepdims):
        n_data = data.shape[0]
        data = data.view(n_data, -1, 3)
        fixed = data[:, fixed_atoms].view(n_data, -1)
        return WhitenFlow(fixed, keepdims=keepdims, whiten_inverse=False)

    def _forward(self, x, *args, **kwargs):
        n_batch = x.shape[0]
        bonds, angles, torsions, x_fixed, dlogp_rel = self._rel_ic(x)
        x_fixed = x_fixed.view(n_batch, -1)
        z_fixed, dlogp_ref = self._whiten(x_fixed)
        dlogp = dlogp_rel + dlogp_ref
        return bonds, angles, torsions, z_fixed, dlogp

    def _inverse(self, bonds, angles, torsions, z_fixed, *args, **kwargs):
        n_batch = z_fixed.shape[0]
        x_fixed, dlogp_ref = self._whiten(z_fixed, inverse=True)
        x_fixed = x_fixed.view(n_batch, -1, 3)
        x, dlogp_rel = self._rel_ic(bonds, angles, torsions, x_fixed, inverse=True)
        dlogp = dlogp_rel + dlogp_ref
        return x, dlogp
