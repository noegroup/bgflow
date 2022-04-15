import pytest
import torch
import warnings
from bgflow.utils import NoClipping, ClipByBatch, ClipByAtom, ClipByValue, ClipBySample


@pytest.fixture
def test_tensor(ctx):
    t = torch.arange(24).reshape(2, 4, 3).to(**ctx)
    t.requires_grad = True
    return t


def test_no_clipping(test_tensor):
    same = NoClipping()(test_tensor)
    assert torch.allclose(test_tensor, same)


def test_clip_by_value(test_tensor):
    clipped = ClipByValue(4.)(test_tensor)
    assert torch.allclose(test_tensor.flatten()[:5], clipped.flatten()[:5])
    assert torch.allclose(torch.ones_like(clipped.flatten()[4:]) * 4., clipped.flatten()[4:])


def test_clip_by_atom(test_tensor):
    clipped = ClipByAtom(4.)(test_tensor)
    assert torch.allclose(test_tensor[0, 0, :], clipped[0, 0, :])
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                continue
            norm = torch.linalg.norm(clipped[i, j], dim=-1)
            assert torch.allclose(norm, torch.tensor(4.).to(norm))


def test_clip_by_sample(test_tensor):
    clipped = ClipBySample(40., n_event_dims=2)(test_tensor)
    assert torch.allclose(test_tensor[0, :, :], clipped[0, :, :])
    assert not torch.allclose(test_tensor[1, :, :], clipped[1, :, :])
    assert torch.allclose(torch.norm(clipped[1].flatten()), torch.tensor(40.).to(clipped))
    should_be_constant = (test_tensor[1, :, :] / clipped[1, :, :])
    assert ((should_be_constant - should_be_constant[0, 0]).abs() < 1e-6).all()


def test_clip_by_batch(test_tensor):
    clipped = ClipByBatch(4.)(test_tensor)
    norm_squared = (clipped * clipped).sum()
    assert torch.allclose(norm_squared, torch.tensor(16.).to(norm_squared))


def test_openmm_clip():
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
    bridge = OpenMMBridge(system, openmm.LangevinIntegrator(300., 0.1, 0.001))

    energy = OpenMMEnergy(bridge=bridge, two_event_dims=False, grad_clipping=NoClipping())
    positions = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 0.6]])
    positions.requires_grad = True
    energy.energy(positions).sum().backward()
    print(torch.linalg.norm(positions.grad), positions.grad)
    assert torch.allclose(positions.grad, -energy.force(positions))
