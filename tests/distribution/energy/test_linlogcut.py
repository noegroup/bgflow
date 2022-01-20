
import pytest
import torch
from bgflow import Energy, LinLogCutEnergy


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
