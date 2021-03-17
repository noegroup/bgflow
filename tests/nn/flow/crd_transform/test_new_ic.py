import torch
import numpy as np

from bgtorch.nn.flow.crd_transform.ic_helper import (
    det2x2,
    det3x3,
    dist_deriv,
    angle_deriv,
    torsion_deriv,
)
from bgtorch.nn.flow.crd_transform.new_ic import (
    ic2xy0_deriv,
    ic2xyz_deriv,
    GlobalInternalCoordinateTransformation,
)

# TODO Floating point precision is brittle!
#      Revision should include numerically more robust
#      implementations - especially for angular values.

torch.set_default_tensor_type = torch.DoubleTensor

N_REPETITIONS = 50


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


def test_angle_deriv(device, dtype, atol=1e-4, rtol=1e-4):
    np.random.seed(123122)
    # check 45 deg angle
    x1 = torch.Tensor([0, 1, 0]).to(device, dtype)
    x2 = torch.Tensor([0, 0, 0]).to(device, dtype)
    x3 = torch.Tensor([1, 1, 0]).to(device, dtype)
    a, J = angle_deriv(x1, x2, x3)
    assert torch.allclose(a, torch.tensor(deg2rad(45.0), device=device, dtype=dtype))
    assert torch.allclose(J, torch.Tensor([-1, 0, 0]).to(device, dtype))

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


def test_torsion_deriv(device, dtype, atol=1e-6, rtol=1e-5):
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


def test_ic2xyz_deriv(device, dtype, atol=1e-5, rtol=1e-4):
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

        assert torch.allclose(x4_new, x4, atol=atol, rtol=rtol)


# TODO: floating point precision is terrible...
def test_global_ic_transform(device, dtype, atol=1e-4, rtol=1e-4):
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
            R = torch.qr(torch.randn(N_SAMPLES, 3, 3, device=device, dtype=dtype))[0][
                :, None
            ]
            x, dlogp_fwd = ic(bonds, angles, torsions, x0, R, inverse=True)
            (
                bonds_recon,
                angles_recon,
                torsions_recon,
                x0_recon,
                R_recon,
                dlogp_inv,
            ) = ic(x)

            failure_message = f"normalize_angles={normalize_angles};"

            # check valid reconstructions
            for name, truth, recon in zip(
                ["bonds", "angles", "torsions", "x0", "R"],
                [bonds, angles, torsions, x0, R],
                [bonds_recon, angles_recon, torsions_recon, x0_recon, R_recon],
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
            R = torch.qr(torch.randn(N_SAMPLES, 3, 3, device=device, dtype=dtype))[0][
                :, None
            ]
            R_noise = torch.qr(
                torch.randn(N_SAMPLES, 3, 3, device=device, dtype=dtype)
            )[0][:, None]

            names = ["bonds", "angles", "torsions", "x0", "R"]
            orig = [bonds, angles, torsions, x0, R]
            noise = [bonds_noise, angles_noise, torsions_noise, x0_noise, R_noise]

            for i, name_noise in enumerate(names):
                noisy_ics = orig[:i] + [noise[i]] + orig[i + 1 :]
                x, _ = ic(*noisy_ics, inverse=True)
                *noisy_ics_recon, _ = ic(x)
                for j, name_recon in enumerate(names):
                    if i != j:
                        assert torch.allclose(
                            orig[j], noisy_ics_recon[j], atol=atol, rtol=rtol
                        ), (failure_message + f"{names[j]} != {name_recon}_recon")
