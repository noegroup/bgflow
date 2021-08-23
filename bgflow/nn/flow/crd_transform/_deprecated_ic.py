import warnings
import numpy as np
import torch

from bgflow.utils.autograd import batch_jacobian
from bgflow.nn.flow.base import Flow


def xyz2ic(x, Z_indices, torsion_cut=None):
    """ Computes internal coordinates from Cartesian coordinates

    Parameters
    ----------
    x : array
        Catesian coordinates
    Z_indices : array
        Internal coordinate index definition. Use -1 to switch off internal coordinates
        when the coordinate system is not fixed.
    mm : energy model
        Molecular model
    torsion_cut : None or array
        If given, defines at which angle to cut the torsions.

    """
    from bgflow.utils.internal_coordinates import dist, angle, torsion

    global_ic = Z_indices.min() < 0
    if global_ic:
        bond_indices = Z_indices[1:, :2]
        angle_indices = Z_indices[2:, :3]
        torsion_indices = Z_indices[3:, :4]
    else:
        bond_indices = Z_indices[:, :2]
        angle_indices = Z_indices[:, :3]
        torsion_indices = Z_indices[:, :4]

    atom_indices = torch.arange(int(3 * (Z_indices.max() + 1))).reshape(-1, 3)
    xbonds = dist(
        x[:, atom_indices[bond_indices[:, 0]]], x[:, atom_indices[bond_indices[:, 1]]]
    )
    xangles = angle(
        x[:, atom_indices[angle_indices[:, 0]]],
        x[:, atom_indices[angle_indices[:, 1]]],
        x[:, atom_indices[angle_indices[:, 2]]],
    )
    xtorsions = torsion(
        x[:, atom_indices[torsion_indices[:, 0]]],
        x[:, atom_indices[torsion_indices[:, 1]]],
        x[:, atom_indices[torsion_indices[:, 2]]],
        x[:, atom_indices[torsion_indices[:, 3]]],
    )

    if torsion_cut is not None:
        xtorsions = torch.where(
            xtorsions < torsion_cut, xtorsions + 2 * np.pi, xtorsions
        )

    # Order ic's by atom
    if global_ic:
        iclist = [xbonds[:, 0:2], xangles[:, 0:1]]
        for i in range(Z_indices.shape[0] - 3):
            iclist += [
                xbonds[:, i + 2 : i + 3],
                xangles[:, i + 1 : i + 2],
                xtorsions[:, i : i + 1],
            ]
    else:
        iclist = []
        for i in range(Z_indices.shape[0]):
            iclist += [
                xbonds[:, i : i + 1],
                xangles[:, i : i + 1],
                xtorsions[:, i : i + 1],
            ]

    ics = torch.cat(iclist, dim=-1)

    return ics


def xyz2ic_log_det_jac(x, Z_indices, eps=1e-10):
    import numpy as np
    from bgflow.utils.internal_coordinates import dist, angle, torsion

    batchsize = x.shape[0]

    atom_indices = np.arange(3 * (Z_indices.max() + 1)).reshape((-1, 3))

    log_det_jac = torch.zeros(batchsize, 1).to(x)

    global_transform = Z_indices.min() < 0
    if global_transform:
        start_rest = 3  # remaining atoms start in row 3

        # 1. bond (input: z axis)
        reference_atom = x[:, atom_indices[Z_indices[1, 0]]]
        other_atom = x[:, atom_indices[Z_indices[1, 1]]]

        x_ = reference_atom[:, 0]
        y_ = reference_atom[:, 1]
        z_ = reference_atom[:, 2]

        # compute first bondlength
        arg = torch.unsqueeze(z_, dim=1)
        reference_atom = torch.stack([x_, y_, arg[:, 0]], dim=-1).unsqueeze(1)
        other_atom = torch.unsqueeze(other_atom, dim=1)
        bondlength = dist(reference_atom, other_atom)

        # This is only 1-dimensional. Summing up directly:
        jac = batch_jacobian(bondlength, arg)
        log_det_jac += jac[:, :, 0]

        # 2. bond/angle (input: x/z axes)
        reference_atom = x[:, atom_indices[Z_indices[2, 0]]]

        x_ = reference_atom[:, 0]
        y_ = reference_atom[:, 1]
        z_ = reference_atom[:, 2]

        arg = torch.stack([x_, z_], dim=-1)
        reference_atom = torch.stack([arg[:, 0], y_, arg[:, 1]], dim=-1).unsqueeze(1)
        other_atom_1 = x[:, atom_indices[Z_indices[2, 1]]].unsqueeze(1)
        other_atom_2 = x[:, atom_indices[Z_indices[2, 2]]].unsqueeze(1)

        bondlength = dist(reference_atom, other_atom_1)
        anglevalue = angle(reference_atom, other_atom_1, other_atom_2)
        out = torch.cat([bondlength, anglevalue], dim=-1)

        # now we have 2x2 matrices
        jac = batch_jacobian(out, arg) + eps * torch.eye(2).unsqueeze(0)
        log_det_jac += torch.slogdet(jac)[-1].unsqueeze(-1)
    else:
        start_rest = 0  # remaining atoms start now

    # 3. all other atoms
    reference_atoms = x[:, atom_indices[Z_indices[start_rest:, 0]]]
    other_atoms_1 = x[:, atom_indices[Z_indices[start_rest:, 1]]]
    other_atoms_2 = x[:, atom_indices[Z_indices[start_rest:, 2]]]
    other_atoms_3 = x[:, atom_indices[Z_indices[start_rest:, 3]]]

    arg = reference_atoms.reshape(-1, 3)

    reference_atoms = arg.reshape(batchsize, -1, 3)
    bondlength = dist(reference_atoms, other_atoms_1)
    anglevalue = angle(reference_atoms, other_atoms_1, other_atoms_2)
    torsionvalue = torsion(reference_atoms, other_atoms_1, other_atoms_2, other_atoms_3)
    out = torch.stack([bondlength, anglevalue, torsionvalue], dim=-1)
    out = out.reshape(-1, 3)

    # 3x3 matrices
    jac = batch_jacobian(out, arg) + eps * torch.eye(3).unsqueeze(0).to(x)
    log_det_jac_ = torch.slogdet(jac)[-1]  # (batchsize * natoms, )
    log_det_jac_ = log_det_jac_.reshape(batchsize, -1)  # (batchsize, natoms)
    log_det_jac_ = log_det_jac_.sum(axis=1, keepdim=True)
    log_det_jac += log_det_jac_

    return log_det_jac


def icmoments(Z_indices, X0=None, torsion_cut=None):
    import numpy as np

    global_ic = Z_indices.min() < 0
    if global_ic:
        dim = 3 * Z_indices.shape[0] - 6
        ntorsions = Z_indices.shape[0] - 3
    else:
        dim = 3 * Z_indices.shape[0]
        ntorsions = Z_indices.shape[0]

    if X0 is not None:
        ics = xyz2ic(X0, Z_indices)
        if global_ic:
            torsions = ics[:, 5::3]
        else:
            torsions = ics[:, 2::3]
        if torsion_cut is None:
            from bgflow.utils.internal_coordinates import torsioncut_mindensity

            torsions_np = torsions.detach().numpy()
            torsion_cut = torch.tensor(
                [torsioncut_mindensity(torsions_np[:, i]) for i in range(ntorsions)]
            )
        # apply torsion cut
        torsion_cut_row = torsion_cut.unsqueeze(0)
        torsions = torch.where(
            torsions < torsion_cut_row, torsions + 2 * np.pi, torsions
        )
        # write torsions back to ics
        if global_ic:
            ics[:, 5::3] = torsions
        else:
            ics[:, 2::3] = torsions
        means = torch.mean(ics, dim=0)
        stds = torch.sqrt(torch.mean((ics - means) ** 2, dim=0))
    else:
        torsion_cut = -np.pi * torch.ones(1, ntorsions)
        means = torch.zeros(1, dim - 6)
        stds = torch.ones(1, dim - 6)
    return means, stds, torsion_cut


def ic2xyz(p1, p2, p3, d14, a124, t1234):
    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2)
    nn = torch.cross(v1, n)
    n_normalized = n / torch.norm(n, dim=1, keepdim=True)
    nn_normalized = nn / torch.norm(nn, dim=1, keepdim=True)

    n_scaled = n_normalized * -torch.sin(t1234)
    nn_scaled = nn_normalized * torch.cos(t1234)

    v3 = n_scaled + nn_scaled
    v3_normalized = v3 / torch.norm(v3, dim=1, keepdim=True)
    v3_scaled = v3_normalized * d14 * torch.sin(a124)

    v1_normalized = v1 / torch.norm(v1, dim=1, keepdim=True)
    v1_scaled = v1_normalized * d14 * torch.cos(a124)

    position = p1 + v3_scaled - v1_scaled

    return position


def decompose_Z_indices(cart_indices, Z_indices):
    import numpy as np

    known_indices = cart_indices
    Z_placed = np.zeros(Z_indices.shape[0])
    Z_indices_decomposed = []
    while np.count_nonzero(Z_placed) < Z_indices.shape[0]:
        Z_indices_cur = []
        for i in range(Z_indices.shape[0]):
            if not Z_placed[i] and np.all(
                [Z_indices[i, j] in known_indices for j in range(1, 4)]
            ):
                Z_indices_cur.append(Z_indices[i])
                Z_placed[i] = 1
        Z_indices_cur = np.array(Z_indices_cur)
        known_indices = np.concatenate([known_indices, Z_indices_cur[:, 0]])
        Z_indices_decomposed.append(Z_indices_cur)

    index2order = np.concatenate(
        [cart_indices] + [Z[:, 0] for Z in Z_indices_decomposed]
    )

    return Z_indices_decomposed, index2order


def ics2xyz_local_log_det_jac_batchexpand(ics, Z_indices, index2zorder, xyz, eps=1e-10):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian

    Parameters
    ----------
    ics : array (batchsize x dim)
        IC matrix flattened by atom to place (bond1, angle1, torsion1, bond2, angle2, torsion2, ...)

    """
    batchsize = ics.shape[0]
    natoms_to_place = Z_indices.shape[0]

    # reshape atoms into the batch
    p1s = xyz[:, index2zorder[Z_indices[:, 1]]].reshape(batchsize * natoms_to_place, 3)
    p2s = xyz[:, index2zorder[Z_indices[:, 2]]].reshape(batchsize * natoms_to_place, 3)
    p3s = xyz[:, index2zorder[Z_indices[:, 3]]].reshape(batchsize * natoms_to_place, 3)
    ics_ = ics.reshape(batchsize * natoms_to_place, 3)

    # operation to differentiate: compute new xyz's given distances, angles, torsions
    newpos_batchexpand = ic2xyz(p1s, p2s, p3s, ics_[:, 0:1], ics_[:, 1:2], ics_[:, 2:3])
    newpos = newpos_batchexpand.reshape(batchsize, natoms_to_place, 3)

    # compute derivatives
    jac = batch_jacobian(newpos_batchexpand, ics_)
    log_det_jac_batchexpand = torch.slogdet(jac)[-1]

    # reshape atoms again out of batch and sum over the log det jacobians
    log_det_jac = log_det_jac_batchexpand.reshape(batchsize, natoms_to_place)
    log_det_jac = torch.sum(log_det_jac, dim=1, keepdim=True)

    return newpos, log_det_jac


def ics2xyz_local_log_det_jac_decomposed(
    all_ics, all_Z_indices, cartesian_xyz, index2order, eps=1e-10
):
    """
    Parameters
    ----------
    all_ics : Tensor (batchsize, 3*nICatoms)
        Tensor with all internal coordinates to be placed, in the order as they are placed in all_Z_indices
    all_Z_indices : list of Z index arrays.
        All atoms in one array are placed independently given the atoms that have been placed before
    cartesian_xyz : Tensor (batchsize, nCartAtoms, 3)
        Start here with the positions of all Cartesian atoms
    index2order : array
        map from atom index to the order of placement. The order of placement is first all Cartesian atoms
        and then in the order of np.vstack(all_Z_indices)[:, 0]

    """
    batchsize = all_ics.shape[0]
    log_det_jac_tot = torch.zeros(batchsize, 1).to(all_ics)
    xyz = cartesian_xyz
    istart = 0
    for Z_indices in all_Z_indices:
        ics = all_ics[:, 3 * istart : 3 * (istart + Z_indices.shape[0])]
        newpos, log_det_jac = ics2xyz_local_log_det_jac_batchexpand(
            ics, Z_indices, index2order, xyz, eps=eps
        )
        xyz = torch.cat([xyz, newpos], axis=1)
        log_det_jac_tot += log_det_jac
        istart += Z_indices.shape[0]
    return xyz, log_det_jac_tot


def periodic_angle_loss(angles):
    """ Penalizes angles outside the range [-pi, pi]

    Use this as an energy loss to avoid violating invertibility in internal coordinate transforms.
    non-unique reconstruction of angles. Computes

        L = (a-pi) ** 2 for a > pi
        L = (a+pi) ** 2 for a < -pi

    and returns the sum over all angles.

    """
    zero = torch.zeros(1, 1).to(angles)
    positive_loss = torch.sum(
        torch.where(angles > np.pi, angles - np.pi, zero) ** 2, dim=-1, keepdim=True
    )
    negative_loss = torch.sum(
        torch.where(angles < -np.pi, angles + np.pi, zero) ** 2, dim=-1, keepdim=True
    )
    return positive_loss + negative_loss


class MixedCoordinateTransform(Flow):
    """ Conversion between Cartesian coordinates and whitened Cartesian / whitened internal coordinates """

    def __init__(
        self,
        cart_atom_indices,
        Z_indices_no_order,
        X0=None,
        X0ic=None,
        remove_dof=6,
        torsion_cut=None,
        jacobian_regularizer=1e-10,
    ):
        """
        Parameters
        ----------
        mm : energy model
            Molecular Model
        cart_atom_indices : array
            Indices of atoms treated as Cartesian, will be whitened with PCA
        ic_atom_indices : list of arrays
            Indices of atoms for which internal coordinates will be computed. Each array defines the Z matrix
            for that IC group.
        X0 : array or None
            Initial coordinates to compute whitening transformations on.
        remove_dof : int
            Number of degrees of freedom to remove from PCA whitening (default is 6 for translation+rotation in 3D)

        """
        warnings.warn(
            "This implementation of the internal coordinate transform is deprecated and will be removed soon. "
            "Please use bgflow.MixedCoordinateTransformation instead. "
            "Note that the new implementation has a different API and is defined in reverse order, "
            "i.e., crd_transform.forward: Cartesian -> ICs.",
            DeprecationWarning,
        )
        super().__init__()

        self.cart_atom_indices = np.array(cart_atom_indices)
        self.cart_indices = np.concatenate(
            [[i * 3, i * 3 + 1, i * 3 + 2] for i in cart_atom_indices]
        )
        self.batchwise_Z_indices, _ = decompose_Z_indices(
            self.cart_atom_indices, Z_indices_no_order
        )
        self.Z_indices = np.vstack(self.batchwise_Z_indices)
        self.dim = 3 * (self.cart_atom_indices.size + self.Z_indices.shape[0])
        self.atom_order = np.concatenate([cart_atom_indices, self.Z_indices[:, 0]])
        self.index2order = np.argsort(self.atom_order)
        self.remove_dof = remove_dof
        self.jacobian_regularizer = jacobian_regularizer

        if X0 is None:
            raise ValueError("Need to specify X0")
        if X0ic is None:
            X0ic = X0

        # Compute PCA transformation on initial data
        from bgflow.nn.flow.crd_transform.pca import _pca

        X0_np = X0.detach().numpy()
        cart_X0mean, cart_Twhiten, cart_Tblacken, std = _pca(
            X0_np[:, self.cart_indices], keepdims=self.cart_indices.size - remove_dof
        )
        self.register_buffer("cart_X0mean", torch.tensor(cart_X0mean))
        self.register_buffer("cart_Twhiten", torch.tensor(cart_Twhiten))
        self.register_buffer("cart_Tblacken", torch.tensor(cart_Tblacken))
        self.register_buffer("std", torch.tensor(std))

        if torch.any(self.std <= 0):
            raise ValueError(
                "Cannot construct whiten layer because trying to keep nonpositive eigenvalues."
            )
        self.register_buffer(
            "pca_log_det_xz", -torch.sum(torch.log(self.std), dim=0, keepdim=True)
        )
        # Compute IC moments for normalization
        ic_means, ic_stds, torsion_cut = icmoments(
            self.Z_indices, X0=X0ic, torsion_cut=torsion_cut
        )
        self.register_buffer("ic_means", ic_means)
        self.register_buffer("ic_stds", ic_stds)
        self.register_buffer("torsion_cut", torsion_cut)

    def _xyz2ic(self, xyz):
        # split off Cartesian coordinates and perform whitening on them
        x_cart = xyz[:, self.cart_indices]
        z_cart_signal = torch.matmul(x_cart - self.cart_X0mean, self.cart_Twhiten)

        # compute and normalize internal coordinates
        z_ics = xyz2ic(xyz, self.Z_indices, torsion_cut=self.torsion_cut)
        z_ics_norm = (z_ics - self.ic_means) / self.ic_stds

        # concatenate
        z = torch.cat([z_cart_signal, z_ics_norm], dim=1)

        # jacobian
        with torch.enable_grad():
            xyz.requires_grad_(True)
            log_det_jac = xyz2ic_log_det_jac(
                xyz, self.Z_indices, eps=self.jacobian_regularizer
            )  # IC part
        log_det_jac += self.pca_log_det_xz  # PCA part

        return z, log_det_jac

    def _ic2xyz(self, z):
        # split off Cartesian block and unwhiten it
        dim_cart_signal = self.cart_indices.size - self.remove_dof
        z_cart_signal = z[:, :dim_cart_signal]
        x_cart = torch.matmul(z_cart_signal, self.cart_Tblacken) + self.cart_X0mean

        # split by atom
        batchsize = z.shape[0]
        xyz = x_cart.reshape(batchsize, self.cart_atom_indices.size, 3)

        # split off Z block
        z_ics_norm = z[:, dim_cart_signal : self.dim - self.remove_dof]
        z_ics = z_ics_norm * self.ic_stds + self.ic_means

        # Compute periodic angle loss
        n_internal = self.dim - dim_cart_signal - self.remove_dof
        angle_idxs = np.arange(n_internal // 3) * 3 + 1
        angles = z_ics[:, angle_idxs]
        torsion_idxs = np.arange(n_internal // 3) * 3 + 2
        torsions_centered = z_ics[:, torsion_idxs] - np.pi - self.torsion_cut
        self.periodic_angle_loss = periodic_angle_loss(angles) + periodic_angle_loss(
            torsions_centered
        )

        # reconstruct remaining atoms using ICs + compute Jacobian
        with torch.enable_grad():
            # xyz.requires_grad_(True)
            z_ics.requires_grad_(True)
            xyz, log_det_jac = ics2xyz_local_log_det_jac_decomposed(
                z_ics,
                self.batchwise_Z_indices,
                xyz,
                self.index2order,
                eps=self.jacobian_regularizer,
            )
        # Add PCA part
        log_det_jac -= self.pca_log_det_xz

        # reorder and concatenate all atom coordinates
        x = xyz[:, self.index2order].reshape(batchsize, -1)

        return x, log_det_jac

    def _forward(self, z, **kwargs):
        x, log_det_jac = self._ic2xyz(z)
        return x, log_det_jac

    def _inverse(self, xyz, **kwargs):
        z, log_det_jac = self._xyz2ic(xyz)
        return z, log_det_jac


def ic2xy0(p1, p2, d14, a124):
    import numpy as np

    # t1234 = tf.Variable(np.array([[90.0 * np.pi / 180.0]], dtype=np.float32))
    t1234 = torch.Tensor([[0.5 * np.pi]])
    p3 = torch.Tensor([[0, -1, 0]])
    return ic2xyz(p1, p2, p3, d14, a124, t1234)


def ics2xyz_global(ics, Z_indices):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian """
    batchsize = ics.shape[0]
    index2zorder = np.argsort(Z_indices[:, 0])
    # Fix coordinate system by placing first three atoms
    xyz = []
    # first atom at 0,0,0
    xyz.append(torch.zeros(batchsize, 3))
    # second atom at 0,0,d
    xyz.append(torch.cat([torch.zeros(batchsize, 2), ics[:, 0:1]], dim=-1))
    # third atom at x,0,z
    xyz.append(
        ic2xy0(
            xyz[index2zorder[Z_indices[2, 1]]],
            xyz[index2zorder[Z_indices[2, 2]]],
            ics[:, 1:2],
            ics[:, 2:3],
        )
    )
    # fill in the rest
    ics2xyz_local(ics[:, 3:], Z_indices[3:], index2zorder, xyz)

    # reorganize indexes
    xyz = [xyz[i] for i in index2zorder]
    return torch.cat(xyz, dim=1)


def ics2xyz_local(ics, Z_indices, index2zorder, xyz):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian

    Parameters
    ----------
    ics : array (batchsize x dim)
        IC matrix flattened by atom to place (bond1, angle1, torsion1, bond2, angle2, torsion2, ...)

    """
    for i in range(Z_indices.shape[0]):
        xyz.append(
            ic2xyz(
                xyz[index2zorder[Z_indices[i, 1]]],
                xyz[index2zorder[Z_indices[i, 2]]],
                xyz[index2zorder[Z_indices[i, 3]]],
                ics[:, 3 * i : 3 * i + 1],
                ics[:, 3 * i + 1 : 3 * i + 2],
                ics[:, 3 * i + 2 : 3 * i + 3],
            )
        )


# def batch_jacobian(out, inp):
#     assert len(out.shape) == 2 and len(inp.shape) == 2
#     assert out.shape[1] == 3 and inp.shape[1] == 3
#     o = out.sum(dim=0)
#     J = torch.stack(
#         torch.autograd.grad([o[0], o[1], o[2]], inp, retain_graph=True, create_graph=True),
#         dim=-1
#     )
#     return J


def ics2xyz_local_log_det_jac(ics, Z_indices, index2zorder, xyz):
    batchsize = ics.shape[0]
    log_det_jac = torch.zeros((batchsize, 1))

    for i in range(Z_indices.shape[0]):
        args = ics[:, 3 * i : 3 * i + 3]
        xyz.append(
            ic2xyz(
                xyz[index2zorder[Z_indices[i, 1]]],
                xyz[index2zorder[Z_indices[i, 2]]],
                xyz[index2zorder[Z_indices[i, 3]]],
                args[:, 0:1],
                args[:, 1:2],
                args[:, 2:3],
            )
        )
        log_det_jac += torch.slogdet(batch_jacobian(xyz[-1], args))[-1].unsqueeze(-1)

    return log_det_jac


# def log_det_jac_lists(ys, xs):
#     from tensorflow.python.ops import gradients as gradient_ops
#
#     batch_dim = xs[0].shape[0]
#     output_dim = ys[0].shape[-1]
#
#     jacs = []
#     for y, x in zip(ys, xs):
#         cols = []
#         for i in range(output_dim):
#             cols.append(gradient_ops.gradients(y[:, i], x)[0])
#         jac = torch.stack(cols, axis=-1)
#         jacs.append(jac)
#
#     log_det = torch.slogdet(jacs)[-1]
#     log_det = torch.sum(log_det, dim=0)
#
#     return log_det


# def ics2xyz_local_log_det_jac_lists(ics, Z_indices, index2zorder, xyz):
#
#     batchsize = ics.shape[0]
#
#     log_det_jac = torch.zeros(batchsize, 1)
#     all_args = []
#     all_outputs = []
#
#     for i in range(Z_indices.shape[0]):
#         all_args.append(torch.cat([ics[:, 3*i:3*i+1],
#                                    ics[:, 3*i+1:3*i+2],
#                                    ics[:, 3*i+2:3*i+3]], axis=-1))
#         xyz.append(ic2xyz(xyz[index2zorder[Z_indices[i, 1]]],
#                           xyz[index2zorder[Z_indices[i, 2]]],
#                           xyz[index2zorder[Z_indices[i, 3]]],
#                           all_args[-1][..., 0:1],
#                           all_args[-1][..., 1:2],
#                           all_args[-1][..., 2:3]))
#         all_outputs.append(xyz[-1])
#
#     log_det_jac = log_det_jac_lists(all_outputs, all_args)
#
#     return log_det_jac


def ics2xyz_global_log_det_jac(ics, Z_indices, global_transform=True):
    import numpy as np

    batchsize = ics.shape[0]

    index2zorder = np.argsort(Z_indices[:, 0])

    xyz = []

    log_det_jac = torch.zeros(batchsize, 1)

    if global_transform:
        # first atom at 0,0,0
        xyz.append(torch.zeros((batchsize, 3)))

        # second atom at 0,0,d
        args = ics[:, 0:1].reshape(batchsize, 1)
        xyz.append(torch.cat([torch.zeros(batchsize, 2), args], dim=-1))
        z = xyz[-1][:, -1:]

        log_det_jac += torch.slogdet(batch_jacobian(z, args))[-1].unsqueeze(-1)

        # third atom at x,0,z
        args = torch.cat([ics[:, 1:2], ics[:, 2:3]], dim=-1)
        xyz.append(
            ic2xy0(
                xyz[index2zorder[Z_indices[2, 1]]],
                xyz[index2zorder[Z_indices[2, 2]]],
                args[..., 0:1],
                args[..., 1:2],
            )
        )
        xz = torch.stack([xyz[-1][:, 0], xyz[-1][:, 2]], dim=-1)
        #  + 1e-6*tf.eye(2, num_columns=2, batch_shape=(1,)
        log_det_jac += torch.slogdet(batch_jacobian(xz, args))[-1].unsqueeze(-1)

    # other atoms
    log_det_jac += ics2xyz_local_log_det_jac(
        ics[:, 3:], Z_indices[3:], index2zorder, xyz
    )

    return log_det_jac


class InternalCoordinatesTransformation(Flow):
    """ Conversion between internal and Cartesian coordinates """

    def __init__(self, Z_indices, Xnorm=None, torsion_cut=None):
        warnings.warn(
            "This implementation of the internal coordinate transform is deprecated and will be removed soon. "
            "Please use the classes from bgflow.nn.flow.crd_transform.ic instead. "
            "Note that the new implementations have different APIs and are defined in reverse order, "
            "i.e., crd_transform.forward: Cartesian -> ICs.",
            DeprecationWarning,
        )
        super().__init__()
        self.dim = Z_indices.shape[0] * 3
        self.Z_indices = Z_indices

        # Compute IC moments for normalization
        self.ic_means, self.ic_stds, self.torsion_cut = icmoments(
            Z_indices, X0=Xnorm, torsion_cut=torsion_cut
        )

    def _xyz2ic(self, xyz):
        # compute and normalize internal coordinates
        ics = xyz2ic(xyz, self.Z_indices, torsion_cut=self.torsion_cut)
        ics_norm = (ics - self.ic_means) / self.ic_stds

        # Jacobian
        log_det_jac = xyz2ic_log_det_jac(xyz, self.Z_indices)

        return ics_norm, log_det_jac

    def _ic2xyz(self, ics):
        # unnormalize
        ics_unnorm = ics * self.ic_stds + self.ic_means
        # reconstruct remaining atoms using ICs
        xyz = ics2xyz_global(ics_unnorm, self.Z_indices)

        # Compute periodic angle loss
        angle_idxs = np.concatenate([[2, 4], np.arange(7, self.dim - 6, 3)])
        angles = ics_unnorm[:, angle_idxs]
        torsion_idxs = np.concatenate([[5], np.arange(8, self.dim - 6, 3)])
        torsions_centered = ics_unnorm[:, torsion_idxs] - np.pi - self.torsion_cut
        self.periodic_angle_loss = periodic_angle_loss(angles) + periodic_angle_loss(
            torsions_centered
        )

        # Jacobian
        log_det_jac = ics2xyz_global_log_det_jac(ics, self.Z_indices)

        # self.angle_loss = keras.layers.Lambda(lambda z: self.z2x(z)[1])(z)
        return xyz, log_det_jac

    def _forward(self, ics, **kwargs):
        x, log_det_jac = self._ic2xyz(ics)
        return x, log_det_jac

    def _inverse(self, xyz, **kwargs):
        ics, log_det_jac = self._xyz2ic(xyz)
        return ics, log_det_jac
