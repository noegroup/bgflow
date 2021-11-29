import pytest
import torch
import numpy as np
from bgflow import XTBEnergy, XTBBridge


@pytest.mark.parametrize("pos_shape", [(1, 3, 3), (1, 9)])
def test_xtb_water(pos_shape, ctx):
    numbers = np.array([8, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, -0.73578586109551],
        [1.44183152868459, 0.00000000000000, 0.36789293054775],
        [-1.44183152868459, 0.00000000000000, 0.36789293054775]],
        **ctx
    )
    target = XTBEnergy(XTBBridge(numbers=numbers), two_event_dims=(pos_shape == (1,3,3)))
    energy = target.energy(positions.reshape(pos_shape))
    force = target.force(positions.reshape(pos_shape))
    assert energy.shape == (1, 1)
    assert force.shape == pos_shape

    expected_energy = torch.tensor(-5.070451354836705, **ctx)
    expected_force = - torch.tensor([
        [6.24500451e-17, - 3.47909735e-17, - 5.07156941e-03],
        [-1.24839222e-03,  2.43536791e-17,  2.53578470e-03],
        [1.24839222e-03, 1.04372944e-17, 2.53578470e-03],
    ], **ctx)
    assert torch.allclose(energy.flatten(), expected_energy.flatten(), atol=1e-5)
    assert torch.allclose(force.flatten(), expected_force.flatten(), atol=1e-5)

