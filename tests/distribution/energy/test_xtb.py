import pytest
import torch
import numpy as np
from bgflow import XTBEnergy, XTBBridge

try:
    import xtb
    xtb_imported = True
except ImportError:
    xtb_imported = False

pytestmark = pytest.mark.skipif(not xtb_imported, reason="Test requires XTB")


@pytest.mark.parametrize("pos_shape", [(1, 3, 3), (1, 9)])
def test_xtb_water(pos_shape, ctx):
    unit = pytest.importorskip("openmm.unit")
    temperature = 300
    numbers = np.array([8, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, -0.73578586109551],
        [1.44183152868459, 0.00000000000000, 0.36789293054775],
        [-1.44183152868459, 0.00000000000000, 0.36789293054775]],
        **ctx
    )
    positions = (positions * unit.bohr).value_in_unit(unit.nanometer)
    target = XTBEnergy(
        XTBBridge(numbers=numbers, temperature=temperature),
        two_event_dims=(pos_shape == (1, 3, 3))
    )
    energy = target.energy(positions.reshape(pos_shape))
    force = target.force(positions.reshape(pos_shape))
    assert energy.shape == (1, 1)
    assert force.shape == pos_shape

    kbt = unit.BOLTZMANN_CONSTANT_kB * temperature * unit.kelvin
    expected_energy = torch.tensor(-5.070451354836705, **ctx) * unit.hartree / kbt
    expected_force = - torch.tensor([
        [6.24500451e-17, - 3.47909735e-17, - 5.07156941e-03],
        [-1.24839222e-03,  2.43536791e-17,  2.53578470e-03],
        [1.24839222e-03, 1.04372944e-17, 2.53578470e-03],
    ], **ctx) * unit.hartree/unit.bohr/(kbt/unit.nanometer)
    assert torch.allclose(energy.flatten(), expected_energy.flatten(), atol=1e-5)
    assert torch.allclose(force.flatten(), expected_force.flatten(), atol=1e-5)


def _eval_invalid(ctx, err_handling):
    pos = torch.zeros(1, 3, 3, **ctx)
    target = XTBEnergy(
        XTBBridge(numbers=np.array([8, 1, 1]), temperature=300, err_handling=err_handling)
    )
    return target.energy(pos), target.force(pos)


def test_xtb_error(ctx):
    from xtb.interface import XTBException
    with pytest.raises(XTBException):
        _eval_invalid(ctx, err_handling="error")


def test_xtb_warning(ctx):
    with pytest.warns(UserWarning, match="Caught exception in xtb"):
        e, f = _eval_invalid(ctx, err_handling="warning")
        assert torch.isinf(e).all()
        assert torch.allclose(f, torch.zeros_like(f))


def test_xtb_ignore(ctx):
    e, f = _eval_invalid(ctx, err_handling="ignore")
    assert torch.isinf(e).all()
    assert torch.allclose(f, torch.zeros_like(f))
