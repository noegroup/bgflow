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

# TODO add local IC transformation
# TODO optimize ic -> xyz transformation
#      using tree-based decomposition
#      of the z-matrix (see old code)


def ic2xyz_deriv(p1, p2, p3, d14, a124, t1234):
    """ computes the xyz coordinates from internal coordinates
        relative to points `p1`, `p2`, `p3` together with its
        jacobian with respect to `p1`.
    """

    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2)
    nn = torch.cross(v1, n)

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
        * (torch.eye(3)[None, :] - outer(v3_normalized, v3_normalized))
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

    t1234 = torch.Tensor([[0.5 * np.pi]])
    p3 = torch.Tensor([[0, -1, 0]])
    xyz, J = ic2xyz_deriv(p1, p2, p3, d14, a124, t1234)
    J = J[..., [0, 1, 2], :][..., [0, 1]]
    return xyz, J


class GlobalInternalCoordinatesTransformation(Flow):
    """ global internal coordinate transformation:
        transforms a system from xyz to ic coordinates and back.
    """

    def __init__(self, z_matrix, normalize_angles=True):
        super().__init__()

        self._z_matrix = z_matrix
        self._inv_z_indices = z_matrix[:, 0].sort().indices

        print(self._inv_z_indices, self._z_matrix[:, 0])

        self._bond_indices = self._z_matrix[1:, :2]
        self._angle_indices = self._z_matrix[2:, :3]
        self._torsion_indices = self._z_matrix[3:, :4]
        self._normalize_angles = normalize_angles

    def _forward(self, x, with_pose=True, *args, **kwargs):
        """ Computes xyz -> ic

            Returns bonds, angles, torsions, x0 and R

            Here x0 is the first point of the system
            and R is the rotation spanned by the first
            three points. Together this determines
            all DOFs
        """

        n_batch = x.shape[0]
        x = x.view(n_batch, -1, 3)

        # compute bonds, angles, torsions
        # together with jacobians (wrt. to diagonal atom)
        bonds, jbonds = dist_deriv(
            x[:, self._bond_indices[:, 0]], x[:, self._bond_indices[:, 1]]
        )
        angles, jangles = angle_deriv(
            x[:, self._angle_indices[:, 0]],
            x[:, self._angle_indices[:, 1]],
            x[:, self._angle_indices[:, 2]],
        )
        torsions, jtorsions = torsion_deriv(
            x[:, self._torsion_indices[:, 0]],
            x[:, self._torsion_indices[:, 1]],
            x[:, self._torsion_indices[:, 2]],
            x[:, self._torsion_indices[:, 3]],
        )

        # aggregate induced volume change
        dlogp = 0.0

        # transforms angles from [-pi, pi] to [0, 1]
        if self._normalize_angles:
            angles = (angles + np.pi) / (2 * np.pi)
            torsions = (torsions + np.pi) / (2 * np.pi)
            dlogp += -np.log(2 * np.pi) * (angles.shape[-1] + torsions.shape[-1])

        # first point necessary to fix first three
        # unconstrained DOFs
        x0 = x[:, [0]]

        # orientation matrix necessary to fix other
        # three unconstrained DOFs
        R = orientation(x[:, 0], x[:, 1], x[:, 2])

        # compute volume change for the transformation
        # of the first three points
        _, _, _, neg_dlogp = self._init_det(
            bonds[..., 0], bonds[..., 1], angles[..., 0]
        )
        dlogp += -neg_dlogp

        # the volume change of the other points
        # is computed the in one step
        j2 = torch.stack(
            [jbonds[..., 2:, :], jangles[..., 1:, :], jtorsions[..., :, :]], dim=-2
        )
        dlogp += det3x3(j2).abs().log().sum(dim=1, keepdim=True)

        return bonds, angles, torsions, x0, R, dlogp

    def _init_det(self, d01, d12, a012):
        """ computes volume change for placing the first two atoms relative x0

            d01: distance between p0 and p1
            d12: distance between p1 and p2
            a012: angle between p0, p1 and p2

            returns p0, p1, p2 and the volume change
        """

        n_batch = d01.shape[0]

        # first point placed in origin
        p0 = torch.zeros(n_batch, 1, 3).to(d01)

        # second point placed in z-axis
        p1 = torch.zeros_like(p0).to(d01)
        p1[..., 2] = d01[:, None]

        # third point placed wrt to p0 and p1
        p2, J = ic2xy0_deriv(p1, p0, d12[:, None, None], a012[:, None, None])

        return p0, p1, p2, det2x2(J[..., [0, 2], :]).abs().log()

    def _inverse(self, bonds, angles, torsions, x0, R, *args, **kwargs):

        dlogp = 0

        # transforms angles from [0, 1] to [-pi, pi]
        if self._normalize_angles:
            angles = angles * (2 * np.pi) - np.pi
            torsions = torsions * (2 * np.pi) - np.pi
            dlogp += np.log(2 * np.pi) * (angles.shape[-1] + torsions.shape[-1])

        # compute position of first three points
        # together with volume change
        p0, p1, p2, dlogp_ = self._init_det(
            bonds[..., 0], bonds[..., 1], angles[..., 0]
        )
        dlogp += dlogp_

        # initially maintain a list of three points
        # which will be updated for each newly computed
        # point poistion
        ps = torch.cat([p0, p1, p2], dim=1)

        # iterate through the z matrix and place
        # points one-by-one
        #
        # TODO this is currently O(N) and could be
        #      sped up significantly if the topology
        #      of the z-matrix is utilized
        #
        #      have a look into the old implementation
        #      as was used in the Science paper!
        for i in range(torsions.shape[-1]):
            ref = self._z_matrix[3 + i, 1:]
            b = bonds[..., i + 2, None]
            a = angles[..., i + 1, None]
            t = torsions[..., i, None]
            ps_ref = ps[:, ref]
            p, J = ic2xyz_deriv(ps_ref[:, 0], ps_ref[:, 1], ps_ref[:, 2], b, a, t)
            dlogp += det3x3(J).abs().log()[:, None]

            ps = torch.cat([ps, p[:, None, :]], dim=1)

        # apply rotation and initial point position
        # to make it a full-rank transformation
        ps = ps @ R.transpose(-1, -2)
        ps = ps + x0

        return ps, dlogp
