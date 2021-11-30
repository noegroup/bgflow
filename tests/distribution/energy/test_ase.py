
import pytest
import torch
from bgflow import ASEBridge, ASEEnergy


try:
    import ase
    ase_imported = True
except ImportError:
    ase_imported = False

pytest.mark.skipif(not ase_imported)


def test_ase_bridge(ctx):
    from ase.build import molecule
    from xtb.ase.calculator import XTB
    water = molecule("H2O")
    water.calc = XTB()
    target = ASEEnergy(ASEBridge(water, 300.))
    pos = torch.tensor(water.positions, **ctx)
    e = target.energy(pos)
    f = target.force(pos)
