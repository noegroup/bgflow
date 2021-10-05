import pytest
import torch
import numpy as np
from bgflow.distribution.energy import LennardJonesPotential
from bgflow.distribution.energy.lennard_jones import lennard_jones_energy_torch


def test_lennard_jones_energy_torch():
    energy_large = lennard_jones_energy_torch(torch.tensor(1e10), eps=1, rm=1)
    energy_min = lennard_jones_energy_torch(torch.tensor(5.), eps=3, rm=5)
    energy_zero = lennard_jones_energy_torch(torch.tensor(2 ** (-1 / 6)), eps=3, rm=1)
    assert torch.allclose(energy_large, torch.tensor(0.))
    assert torch.allclose(energy_min, torch.tensor(-3.))
    assert torch.allclose(energy_zero, torch.tensor(0.), atol=1e-5)


@pytest.mark.parametrize("oscillator", [True, False])
@pytest.mark.parametrize("two_event_dims", [True, False])
def test_lennard_jones_potential(oscillator, two_event_dims):
    eps = 5.

    # 2 particles in 3D
    lj_pot = LennardJonesPotential(
        dim=6, n_particles=2, eps=eps, rm=2.0,
        oscillator=oscillator, oscillator_scale=1.,
        two_event_dims=two_event_dims
    )

    batch_shape = (5, 7)
    data3d = torch.tensor([[[[-1., 0, 0], [1, 0, 0]]]]).repeat(*batch_shape, 1, 1)
    if not two_event_dims:
        data3d = data3d.view(*batch_shape, 6)
    energy3d = torch.tensor([[- eps]]).repeat(*batch_shape)
    if oscillator:
        energy3d += 1
    lj_energy_3d = lj_pot.energy(data3d)
    assert torch.allclose(energy3d[:, None], lj_energy_3d)

    # 3 particles in 2D
    lj_pot = LennardJonesPotential(
        dim=6, n_particles=3, eps=eps, rm=1.0,
        oscillator=oscillator, oscillator_scale=1.,
        two_event_dims=two_event_dims
    )
    h = np.sqrt(0.75)
    data2d = torch.tensor([[[0, 2 / 3 * h], [0.5, -1 / 3 * h], [-0.5, -1 / 3 * h]]], dtype=torch.float)
    if not two_event_dims:
        data2d = data2d.view(-1, 6)
    energy2d = torch.tensor([- 3 * eps])
    if oscillator:
        energy2d += 0.5 * (data2d ** 2).sum()
    lj_energy_2d = lj_pot.energy(data2d)
    assert torch.allclose(energy2d, lj_energy_2d)
    lj_energy2d_np = lj_pot._energy_numpy(data2d)
    assert energy2d[:, None].numpy() == pytest.approx(lj_energy2d_np, abs=1e-6)
