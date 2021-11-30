
import pytest
import torch
from bgflow import ASEBridge, ASEEnergy, XTBBridge, XTBEnergy


try:
    import ase
    import xtb
    ase_and_xtb_imported = True
except ImportError:
    ase_and_xtb_imported = False

pytestmark = pytest.mark.skipif(not ase_and_xtb_imported, reason="Tests require ASE and XTB")


def test_ase_energy(ctx):
    from ase.build import molecule
    from xtb.ase.calculator import XTB
    water = molecule("H2O")
    water.calc = XTB()
    target = ASEEnergy(ASEBridge(water, 300.))
    pos = torch.tensor(0.1*water.positions, **ctx)
    e = target.energy(pos)
    f = target.force(pos)


def test_ase_vs_xtb(ctx):
    # to make sure that unit conversion is the same, etc.
    from ase.build import molecule
    from xtb.ase.calculator import XTB
    water = molecule("H2O")
    water.calc = XTB()
    target1 = ASEEnergy(ASEBridge(water, 300.))
    target2 = XTBEnergy(XTBBridge(water.numbers, 300.))
    pos = torch.tensor(0.1 * water.positions[None, ...], **ctx)
    assert torch.allclose(target1.energy(pos), target2.energy(pos))
    assert torch.allclose(target1.force(pos), target2.force(pos), atol=1e-6)

