import pytest
import torch
from bgflow.distribution import MultiDoubleWellPotential


def test_multi_double_well_potential(ctx):
    target = MultiDoubleWellPotential(dim=4, n_particles=2, a=0.9, b=-4, c=0, offset=4)
    x = torch.tensor([[[2., 0, ], [-2, 0]]], **ctx)
    energy = target.energy(x)
    target.force(x)
    assert torch.allclose(energy, torch.tensor([[0.]], **ctx), atol=1e-5)
