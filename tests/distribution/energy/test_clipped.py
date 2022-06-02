
import pytest
import warnings
import torch
from bgflow import Energy, LinLogCutEnergy, GradientClippedEnergy, DoubleWellEnergy
from bgflow.utils import ClipGradient


class StrongRepulsion(Energy):
    def __init__(self):
        super().__init__([2, 2])

    def _energy(self, x):
        dist = torch.cdist(x, x)
        return (dist ** -12)[..., 0, 1][:, None]


def test_linlogcut(ctx):
    lj = StrongRepulsion()
    llc = LinLogCutEnergy(lj, high_energy=1e3, max_energy=1e10)
    x = torch.tensor([
        [[0., 0.], [0.0, 0.0]],  # > max energy
        [[0., 0.], [0.0, 0.3]],  # < max_energy, > high_energy
        [[0., 0.], [0.0, 1.]],  # < high_energy
    ], **ctx)
    raw = lj.energy(x)[:, 0]
    cut = llc.energy(x)[:, 0]

    # first energy is clamped
    assert not (raw <= 1e10).all()
    assert (cut <= 1e10).all()
    assert cut[0].item() == pytest.approx(1e10, abs=1e-5)
    # second energy is softened, but force points in the right direction
    assert 1e3 < cut[1].item() < 1e10
    assert llc.force(x)[1][1, 1] > 0.0
    assert llc.force(x)[1][0, 1] < 0.0
    # third energy is unchanged
    assert torch.allclose(raw[2], cut[2], atol=1e-5)


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
    energy = GradientClippedEnergy(energy, grad_clipping).to(**ctx)
    positions = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.6]]).to(**ctx)
    positions.requires_grad = True
    force = energy.force(positions)
    energy.energy(positions).sum().backward()
    #force = torch.tensor([[-1908.0890,  -3816.1780, -11448.5342, 1908.0890, 3816.1780, 11448.5342]]).to(**ctx)
    return positions.grad, force


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


def test_openmm_clip_no_grad(ctx):
    energy = GradientClippedEnergy(
        energy=DoubleWellEnergy(2),
        gradient_clipping=ClipGradient(clip=1.0, norm_dim=1)
    )
    x = torch.randn(12,2).to(**ctx)
    x.requires_grad = False
    energy.energy(x)
