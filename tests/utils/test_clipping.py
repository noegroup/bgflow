import pytest
import torch
import warnings
from bgflow.utils import ClipGradient


def torch_example(grad_clipping, ctx):
    positions = torch.arange(6).reshape(2, 3).to(**ctx)
    positions.requires_grad = True
    positions = grad_clipping.to(**ctx)(positions)
    (0.5 * positions ** 2).sum().backward()
    return positions.grad


def test_clip_by_val(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=1)
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        torch.tensor([[0., 1., 2.], [3., 3., 3.]], **ctx)
    )


def test_clip_by_atom(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=3)
    norm2 = torch.linalg.norm(torch.arange(3, 6, **ctx)).item()
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        torch.tensor([[0., 1., 2.], [3/norm2*3, 4/norm2*3, 5/norm2*3]], **ctx)
    )


def test_clip_by_batch(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=-1)
    norm2 = torch.linalg.norm(torch.arange(6, **ctx)).item()
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        (torch.arange(6, **ctx) / norm2 * 3.).reshape(2, 3)
    )


def openmm_example(grad_clipping, ctx):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", DeprecationWarning
            )  # ignore warnings inside OpenMM
            from simtk import openmm, unit
    except ImportError:
        pytest.skip("Test requires OpenMM.")

    system = openmm.System()
    system.addParticle(1.)
    system.addParticle(2.)
    nonbonded = openmm.NonbondedForce()
    nonbonded.addParticle(0.0, 1.0, 2.0)
    nonbonded.addParticle(0.0, 1.0, 2.0)
    system.addForce(nonbonded)

    from bgflow import OpenMMEnergy, OpenMMBridge
    bridge = OpenMMBridge(system, openmm.LangevinIntegrator(300., 0.1, 0.001), n_workers=1)

    energy = OpenMMEnergy(bridge=bridge, two_event_dims=False)
    positions = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.6]]).to(**ctx)
    positions.requires_grad = True
    positions = grad_clipping.to(**ctx)(positions)
    energy.energy(positions).sum().backward()
    force = energy.force(positions)
    return positions.grad, force
    # force ([[-1908.0890,  -3816.1780, -11448.5342, 1908.0890, 3816.1780, 11448.5342]])


def test_openmm_clip_by_value(ctx):
    grad_clipping = ClipGradient(clip=3000.0, norm_dim=1)
    grad, force = openmm_example(grad_clipping, ctx)
    expected = - torch.as_tensor([[-1908.0890,  -3000., -3000., 1908.0890, 3000., 3000.]], **ctx)
    assert torch.allclose(grad.flatten(), expected, atol=1e-3)


def test_openmm_clip_by_atom(ctx):
    grad_clipping = ClipGradient(clip=torch.as_tensor([3000.0, 1.0]), norm_dim=3)
    grad, force = openmm_example(grad_clipping, ctx)
    norm_ratio = torch.linalg.norm(grad[..., :3], dim=-1).item()
    assert norm_ratio == pytest.approx(3000.)
    assert torch.allclose(grad[..., :3] / 3000., - grad[..., 3:], atol=1e-6)


def test_openmm_clip_by_batch(ctx):
    grad_clipping = ClipGradient(clip=1.0, norm_dim=-1)
    grad, force = openmm_example(grad_clipping, ctx)
    ratio = force / grad
    assert torch.allclose(ratio, ratio[0, 0] * torch.ones_like(ratio))
    assert torch.linalg.norm(grad).item() == pytest.approx(1.)
