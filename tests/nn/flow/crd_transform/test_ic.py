import torch
import numpy as np
import pytest

from bgflow.nn.flow.crd_transform.ic_helper import (
    det2x2,
    det3x3,
    dist_deriv,
    angle_deriv,
    torsion_deriv,
    ic2xy0_deriv,
    ic2xyz_deriv,
    init_xyz2ics,
    init_ics2xyz,
)
from bgflow.nn.flow.crd_transform.ic import (
    GlobalInternalCoordinateTransformation,
    RelativeInternalCoordinateTransformation,
    MixedCoordinateTransformation,
    decompose_z_matrix,
)


# TODO Floating point precision is brittle!
#      Revision should include numerically more robust
#      implementations - especially for angular values.

TOLERANCES = {
    torch.device("cuda:0"): {torch.float32: (1e-2, 1e-3), torch.float64: (1e-6, 1e-6)},
    torch.device("cpu"): {torch.float32: (1e-4, 1e-4), torch.float64: (1e-7, 1e-7)}
}


N_REPETITIONS = 50


@pytest.fixture()
def alanine_ics():
    """Examplary z-matrix, fixed atoms, and positions for capped alanine."""
    rigid_block = np.array([6, 8, 9, 10, 14])

    relative_z_matrix = np.array(
        [
            [0, 1, 4, 6],
            [1, 4, 6, 8],
            [2, 1, 4, 0],
            [3, 1, 4, 0],
            [4, 6, 8, 14],
            [5, 4, 6, 8],
            [7, 6, 8, 4],
            [11, 10, 8, 6],
            [12, 10, 8, 11],
            [13, 10, 8, 11],
            [15, 14, 8, 16],
            [16, 14, 8, 6],
            [17, 16, 14, 15],
            [18, 16, 14, 8],
            [19, 18, 16, 14],
            [20, 18, 16, 19],
            [21, 18, 16, 19],
        ]
    )

    global_z_matrix = np.array(
        [
            [0, -1, -1, -1],
            [1, 0, -1, -1],
            [2, 1, 0, -1],
            [3, 1, 0, 2],
            [4, 1, 0, 2],
            [5, 4, 1, 0],
            [6, 4, 1, 5],
            [7, 6, 4, 1],
            [8, 6, 4, 7],
            [9, 8, 6, 4],
            [10, 8, 6, 9],
            [14, 8, 6, 9],
            [11, 10, 8, 6],
            [12, 10, 8, 11],
            [13, 10, 8, 11],
            [15, 14, 8, 6],
            [16, 14, 8, 15],
            [17, 16, 14, 15],
            [18, 16, 17, 14],
            [19, 18, 16, 14],
            [20, 18, 19, 16],
            [21, 18, 20, 16],
        ]
    )

    xyz = np.array(
        [
            [1.375, 1.25, 1.573],
            [1.312, 1.255, 1.662],
            [1.327, 1.306, 1.493],
            [1.377, 1.143, 1.549],
            [1.511, 1.31, 1.618],
            [1.606, 1.236, 1.63],
            [1.523, 1.441, 1.633],
            [1.445, 1.5, 1.607],
            [1.645, 1.515, 1.667],
            [1.703, 1.459, 1.74],
            [1.73, 1.53, 1.54],
            [1.792, 1.619, 1.554],
            [1.78, 1.439, 1.508],
            [1.663, 1.555, 1.457],
            [1.618, 1.646, 1.734],
            [1.509, 1.703, 1.709],
            [1.715, 1.705, 1.809],
            [1.798, 1.653, 1.831],
            [1.703, 1.847, 1.852],
            [1.801, 1.871, 1.892],
            [1.674, 1.911, 1.768],
            [1.631, 1.858, 1.933],
        ]
    )

    return relative_z_matrix, global_z_matrix, rigid_block, xyz.reshape(1, -1)


def rad2deg(x):
    return x * 180.0 / np.pi


def deg2rad(x):
    return x * np.pi / 180.0


# def test_outer(device, dtype, atol=1e-6, rtol=1e-5):
#    for _ in range(N_REPETITIONS):
#        x, y = torch.Tensor(2, 5, 7, 3).to(device, dtype).normal_()
#        A = outer(x, y).view(-1)
#        B = []
#        for i in range(5):
#            for j in range(7):
#                for k in range(3):
#                    for l in range(3):
#                        B.append(x[i, j, k] * y[i, j, l])
#        B = torch.Tensor(B).to(device, dtype)
#        assert torch.allclose(A, B, atol=atol, rtol=rtol)


def test_det2x2(device, dtype, atol=1e-6, rtol=1e-5):
    for _ in range(N_REPETITIONS):
        x = torch.Tensor(7, 5, 3, 2, 2).to(device, dtype).normal_()
        assert torch.allclose(det2x2(x), x.det(), atol=atol, rtol=rtol)


def test_det3x3(device, dtype, atol=5e-6, rtol=5e-5):
    for _ in range(N_REPETITIONS):
        x = torch.Tensor(7, 5, 3, 3, 3).to(device, dtype).normal_()
        if not torch.allclose(det3x3(x), x.det(), atol=atol, rtol=rtol):
            print(det3x3(x) - x.det())
        assert torch.allclose(det3x3(x), x.det(), atol=atol, rtol=rtol)


def test_dist_deriv(device, dtype, atol=1e-6, rtol=1e-5):
    x1 = torch.Tensor([0, 0, 0]).to(device, dtype)
    x2 = torch.Tensor([1, 1, 0]).to(device, dtype)
    d, J = dist_deriv(x1, x2)
    sqrt2 = torch.Tensor([2]).to(device, dtype).sqrt()
    assert torch.allclose(d, sqrt2, atol=atol, rtol=rtol)
    assert torch.allclose(J, -x2 / sqrt2)


def test_angle_deriv(device, dtype):
    atol = 1e-2 if dtype is torch.float32 and device is torch.device("cuda:0") else 1e-4
    rtol = 1e-3 if dtype is torch.float32 else 1e-5
    atol, rtol = TOLERANCES[device][dtype]
    np.random.seed(123122)
    # check 45 deg angle
    x1 = torch.Tensor([0, 1, 0]).to(device, dtype)
    x2 = torch.Tensor([0, 0, 0]).to(device, dtype)
    x3 = torch.Tensor([1, 1, 0]).to(device, dtype)
    a, J = angle_deriv(x1, x2, x3)
    assert torch.allclose(a, torch.tensor(deg2rad(45.0), device=device, dtype=dtype))
    assert torch.allclose(J, torch.Tensor([-1, 0, 0]).to(device, dtype), atol=atol)

    # check random angle
    for i in range(N_REPETITIONS):

        # random reference angle
        # TODO: more stable angle derivatives
        a_ref = np.random.uniform(
            1e-2, np.pi - 1e-2
        )  # prevent angles with numerical issues
        x1 = (
            torch.Tensor([np.cos(a_ref), np.sin(a_ref), 0])
            .to(device, dtype)
            .requires_grad_(True)
        )

        # construct system in standard basis
        x2 = torch.Tensor([0, 0, 0]).to(device, dtype)
        x3 = torch.Tensor([1, 0, 0]).to(device, dtype)

        # apply random rotation to system
        R = torch.tensor(
            np.linalg.qr(np.random.uniform(size=(3, 3)))[0], dtype=dtype, device=device
        )
        x1, x2, x3 = (x @ R for x in (x1, x2, x3))

        a, J = angle_deriv(x1, x2, x3)

        # compute Jacobian with autograd
        J_ref = torch.autograd.grad(a.sum(), x1)[0]

        assert torch.allclose(
            a, torch.tensor(a_ref, dtype=dtype, device=device), atol=atol, rtol=rtol
        )
        assert torch.allclose(J, J_ref, atol=atol, rtol=rtol)


def test_torsion_deriv(device, dtype):
    atol, rtol = TOLERANCES[device][dtype]
    np.random.seed(202422)
    for i in range(N_REPETITIONS):

        # random reference angle
        a_ref = np.random.uniform(0, np.pi)

        # construct system in standard basis
        x1 = (
            torch.Tensor([np.cos(a_ref), np.sin(a_ref), 1])
            .to(device, dtype)
            .requires_grad_(True)
        )
        x2 = torch.Tensor([0, 0, 1]).to(device, dtype).unsqueeze(0)
        x3 = torch.Tensor([0, 0, -1]).to(device, dtype).unsqueeze(0)
        x4 = torch.Tensor([1, 0, -1]).to(device, dtype).unsqueeze(0)

        # apply random rotation to system
        R = torch.Tensor(np.linalg.qr(np.random.uniform(size=(3, 3)))[0]).to(
            device, dtype
        )
        x1, x2, x3, x4 = (x @ R for x in (x1, x2, x3, x4))

        # torsion angle should be invariant under rotation
        a, J = torsion_deriv(x1, x2, x3, x4)

        # compute Jacobian with autograd
        J_ref = torch.autograd.grad(a.sum(), x1)[0]

        assert torch.allclose(
            a, torch.tensor(a_ref, device=device, dtype=dtype), atol=atol, rtol=rtol
        )
        assert torch.allclose(J, J_ref, atol=atol, rtol=rtol)


def test_ic2xyz_deriv(device, dtype): 
    atol, rtol = TOLERANCES[device][dtype]
    np.random.seed(202982)
    for i in range(N_REPETITIONS):

        d12 = torch.tensor(np.random.uniform(0.5, 1.5), device=device, dtype=dtype)
        d23 = torch.tensor(np.random.uniform(0.5, 1.5), device=device, dtype=dtype)
        a = torch.tensor(np.random.uniform(0, np.pi), device=device, dtype=dtype)

        # first point placed in origin
        x1 = torch.zeros(1, 1, 3, device=device, dtype=dtype)

        # second point placed in z-axis
        x2 = torch.zeros_like(x1)
        x2[..., 2] = d12

        # third point placed wrt to p0 and p1
        x3, _ = ic2xy0_deriv(x1, x2, d23, a)

        # fourth point placed randomly
        x4 = torch.Tensor(1, 1, 3).to(device, dtype).normal_()

        # compute ic
        d, Jd = dist_deriv(x4, x3)
        a, Ja = angle_deriv(x4, x3, x2)
        t, Jt = torsion_deriv(x4, x3, x2, x1)

        # reconstruct 4th point
        x4_new, J = ic2xyz_deriv(x3, x2, x1, d, a, t)

        assert torch.allclose(x4_new, x4, atol=atol)


# TODO: floating point precision is terrible...
@pytest.mark.filterwarnings("ignore:singular division")
def test_global_ic_transform(device, dtype):
    atol, rtol = TOLERANCES[device][dtype]
    torch.manual_seed(1)

    if dtype == torch.float32:
        atol = 1e-3
        rtol = 1e-3
    elif dtype == torch.float64:
        atol = 1e-5
        rtol = 1e-4

    N_SAMPLES = 1
    N_BONDS = 4
    N_ANGLES = 3
    N_TORSIONS = 2
    N_PARTICLES = 5

    _Z_MATRIX = np.array(
        [[0, -1, -1, -1], [1, 0, -1, -1], [2, 1, 0, -1], [3, 2, 1, 0], [4, 3, 2, 1]]
    )

    for _ in range(N_REPETITIONS):

        for normalize_angles in [True, False]:

            ic = GlobalInternalCoordinateTransformation(
                _Z_MATRIX, normalize_angles=normalize_angles
            )

            # Test ic -> xyz -> ic reconstruction
            bonds = torch.randn(N_SAMPLES, N_BONDS, device=device, dtype=dtype).exp()
            angles = torch.rand(N_SAMPLES, N_ANGLES, device=device, dtype=dtype)
            torsions = torch.rand(N_SAMPLES, N_TORSIONS, device=device, dtype=dtype)

            if not normalize_angles:
                angles *= np.pi
                torsions = (2 * torsions - 1) * np.pi

            x0 = torch.randn(N_SAMPLES, 1, 3, device=device, dtype=dtype)

            alpha = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            beta = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            gamma = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            if not normalize_angles:
                alpha = alpha * 2 * np.pi - np.pi
                beta = 2 * beta - 1
                gamma = gamma * 2 * np.pi - np.pi
            orientation = torch.stack([alpha, beta, gamma], dim=-1)

            x, dlogp_fwd = ic(bonds, angles, torsions, x0, orientation, inverse=True)
            (
                bonds_recon,
                angles_recon,
                torsions_recon,
                x0_recon,
                orientation_recon,
                dlogp_inv,
            ) = ic(x)

            failure_message = f"normalize_angles={normalize_angles};"

            # check valid reconstructions
            for name, truth, recon in zip(
                ["bonds", "angles", "torsions", "x0", "orientation"],
                [bonds, angles, torsions, x0, orientation],
                [
                    bonds_recon,
                    angles_recon,
                    torsions_recon,
                    x0_recon,
                    orientation_recon,
                ],
            ):
                assert torch.allclose(truth, recon, atol=atol, rtol=rtol), (
                    failure_message + f"{name} != {name}_recon;"
                )
            assert torch.allclose(
                (dlogp_fwd + dlogp_inv).exp(),
                torch.ones_like(dlogp_fwd),
                atol=1e-3,
                rtol=1.0,
            ), failure_message

            # Test xyz -> ic -> xyz reconstruction
            x = torch.randn(N_SAMPLES, N_PARTICLES * 3, device=device, dtype=dtype)

            *ics, dlogp_fwd = ic(x)
            x_recon, dlogp_inv = ic(*ics, inverse=True)

            assert torch.allclose(x, x_recon, atol=atol, rtol=rtol), failure_message
            assert torch.allclose(
                (dlogp_fwd + dlogp_inv).exp(),
                torch.ones_like(dlogp_fwd),
                atol=1e-3,
                rtol=1.0,
            ), failure_message

            # Test IC independence
            bonds, bonds_noise = torch.randn(
                2, N_SAMPLES, N_BONDS, device=device, dtype=dtype
            ).exp()
            angles, angles_noise = torch.rand(
                2, N_SAMPLES, N_ANGLES, device=device, dtype=dtype
            )
            torsions, torsions_noise = torch.rand(
                2, N_SAMPLES, N_TORSIONS, device=device, dtype=dtype
            )
            x0, x0_noise = torch.randn(2, N_SAMPLES, 1, 3, device=device, dtype=dtype)

            alpha = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            beta = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            gamma = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            if not normalize_angles:
                alpha = alpha * 2 * np.pi - np.pi
                beta = 2 * beta - 1
                gamma = gamma * 2 * np.pi - np.pi
            orientation = torch.stack([alpha, beta, gamma], dim=-1)

            alpha_noise = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            beta_noise = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            gamma_noise = torch.rand(N_SAMPLES, device=device, dtype=dtype)
            if not normalize_angles:
                alpha_noise = alpha_noise * 2 * np.pi - np.pi
                beta_noise = 2 * beta_noise - 1
                gamma_noise = gamma_noise * 2 * np.pi - np.pi
            orientation_noise = torch.stack(
                [alpha_noise, beta_noise, gamma_noise], dim=-1
            )

            names = ["bonds", "angles", "torsions", "x0", "orientation"]
            orig = [bonds, angles, torsions, x0, orientation]
            noise = [
                bonds_noise,
                angles_noise,
                torsions_noise,
                x0_noise,
                orientation_noise,
            ]

            for i, name_noise in enumerate(names):
                noisy_ics = orig[:i] + [noise[i]] + orig[i + 1 :]
                x, _ = ic(*noisy_ics, inverse=True)
                *noisy_ics_recon, _ = ic(x)
                for j, name_recon in enumerate(names):
                    if i != j:
                        assert torch.allclose(
                            orig[j], noisy_ics_recon[j], atol=atol, rtol=rtol
                        ), (failure_message + f"{names[j]} != {name_recon}_recon")


def test_global_ic_properties(ctx):
    zmat = np.array(
        [[0, -1, -1, -1], [1, 0, -1, -1], [2, 1, 0, -1], [3, 2, 1, 0], [4, 3, 2, 1]]
    )
    dim = 15
    batch_dim = 10

    ic = GlobalInternalCoordinateTransformation(zmat).to(**ctx)
    ics = ic.forward(torch.randn(batch_dim, dim, **ctx))
    assert (zmat[3:] == ic.z_matrix).all()
    assert len(ic.fixed_atoms) == 0
    assert ics[0].shape == (batch_dim, ic.dim_bonds)
    assert ics[1].shape == (batch_dim, ic.dim_angles)
    assert ics[2].shape == (batch_dim, ic.dim_torsions)
    assert ics[3].shape == (batch_dim, 1, 3)
    assert ics[4].shape == (batch_dim, 3)
    assert ic.dim_fixed == 0
    assert ic.normalize_angles
    assert (ic.bond_indices == zmat[1:, :2]).all()
    assert (ic.angle_indices == zmat[2:, :3]).all()
    assert (ic.torsion_indices == zmat[3:, :]).all()


def test_relative_ic_properties(ctx):
    zmat = np.array([[3, 2, 1, 0], [4, 3, 2, 1]])
    fixed_atoms = np.array([0, 1, 2])
    dim = 15
    batch_dim = 10

    ic = RelativeInternalCoordinateTransformation(zmat, fixed_atoms).to(**ctx)
    ics = ic.forward(torch.randn(batch_dim, dim, **ctx))
    assert np.allclose(zmat, ic.z_matrix)
    assert np.allclose(fixed_atoms, ic.fixed_atoms)
    assert ics[0].shape == (batch_dim, ic.dim_bonds)
    assert ics[1].shape == (batch_dim, ic.dim_angles)
    assert ics[2].shape == (batch_dim, ic.dim_torsions)
    assert ics[3].shape == (batch_dim, ic.dim_fixed)
    assert ic.normalize_angles
    assert (ic.bond_indices == zmat[:, :2]).all()
    assert (ic.angle_indices == zmat[:, :3]).all()
    assert (ic.torsion_indices == zmat[:, :]).all()


def test_mixed_ic_properties(ctx):
    zmat = np.array([[3, 2, 1, 0], [4, 3, 2, 1]])
    fixed_atoms = np.array([0, 1, 2])
    dim = 15
    batch_dim = 10
    data = torch.randn(1000, dim, **ctx)

    ic = MixedCoordinateTransformation(data, zmat, fixed_atoms, keepdims=6).to(**ctx)
    ics = ic.forward(torch.randn(batch_dim, dim, **ctx))
    assert np.allclose(zmat, ic.z_matrix)
    assert np.allclose(fixed_atoms, ic.fixed_atoms)
    assert ics[0].shape == (batch_dim, ic.dim_bonds)
    assert ics[1].shape == (batch_dim, ic.dim_angles)
    assert ics[2].shape == (batch_dim, ic.dim_torsions)
    assert ics[3].shape == (batch_dim, ic.dim_fixed)
    assert ic.normalize_angles
    assert (ic.bond_indices == zmat[:, :2]).all()
    assert (ic.angle_indices == zmat[:, :3]).all()
    assert (ic.torsion_indices == zmat[:, :]).all()


def test_decompose_z_matrix(alanine_ics):
    z_matrix, _, rigid_block, _ = alanine_ics
    blocks, index2atom, atom2index, index2order = decompose_z_matrix(
        z_matrix, rigid_block
    )
    blocks_cat = np.concatenate(blocks, axis=0)
    row_order = np.argsort(blocks_cat[:, 0])
    assert (blocks_cat[row_order] == z_matrix).all()
    assert (z_matrix[index2order[np.arange(len(z_matrix))]] == blocks_cat).all()
    assert (
        index2atom[atom2index[np.arange(len(z_matrix))]] == np.arange(len(z_matrix))
    ).all()
    # makes sure all atoms can be reconstructed in this order
    placed = rigid_block
    for block in blocks:
        required_atoms = block[:, 2:].flatten()
        assert np.all(np.isin(required_atoms, placed))
        placed = np.concatenate([placed, block.flatten()])


def test_global_ic_inversion(ctx, alanine_ics):
    tol = 1e-3 if ctx["dtype"] is torch.float32 else 1e-5
    _, z_matrix, _, positions = alanine_ics
    ic = GlobalInternalCoordinateTransformation(z_matrix).to(**ctx)
    positions = torch.tensor(positions, **ctx)
    *out, dlogp = ic.forward(positions)
    positions2, dlogp2 = ic.forward(*out, inverse=True)
    assert torch.allclose(positions, positions2, atol=tol)
    assert torch.allclose(dlogp, -dlogp2, atol=tol)


def test_relative_ic_inversion(ctx, alanine_ics):
    z_matrix, _, rigid_block, positions = alanine_ics
    ic = RelativeInternalCoordinateTransformation(z_matrix, rigid_block).to(**ctx)
    positions = torch.tensor(positions, **ctx)
    *out, dlogp = ic.forward(positions)
    positions2, dlogp2 = ic.forward(*out, inverse=True)
    assert torch.allclose(positions, positions2)
    assert torch.allclose(dlogp, -dlogp2)


def test_mixed_ic_inversion(ctx, alanine_ics):
    z_matrix, _, rigid_block, positions = alanine_ics
    positions = torch.tensor(positions, **ctx)
    data = positions.repeat(100, 1) + torch.randn(100, len(positions), **ctx)
    ic = MixedCoordinateTransformation(data, z_matrix, rigid_block, keepdims=5).to(
        **ctx
    )
    *out, dlogp = ic.forward(positions)
    positions2, dlogp2 = ic.forward(*out, inverse=True)
    assert torch.allclose(positions, positions2)
    assert torch.allclose(dlogp, -dlogp2)
