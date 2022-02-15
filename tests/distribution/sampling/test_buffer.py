
import torch
import numpy as np
import pytest

from bgflow import (
    MetropolizedReplayBuffer, ReplayBufferHDF5File, ReplayBufferHDF5Reporter,
    NormalDistribution, ProductDistribution
)
from bgflow.utils import pack_tensor_in_tuple
from bgflow.utils.types import as_numpy


@pytest.mark.parametrize("use_product", [True, False])
@pytest.mark.parametrize("with_reporter", [True, False])
def test_replay_buffer_sampling(use_product, with_reporter, ctx, tmpdir):
    target = NormalDistribution(dim=2).to(**ctx)
    proposal = NormalDistribution(dim=2, mean=1.0*torch.ones(2, **ctx)).to(**ctx)
    if use_product:
        target = ProductDistribution([target, target])
        proposal = ProductDistribution([proposal, proposal])

    def target_prob_higher(*x):
        return target.energy(*x).mean() < proposal.energy(*x).mean()

    assert target_prob_higher(*pack_tensor_in_tuple(target.sample(100)))
    assert not target_prob_higher(*pack_tensor_in_tuple(proposal.sample(100)))

    if with_reporter:
        reporter = ReplayBufferHDF5Reporter(tmpdir/"test.h5", "w", write_buffer_interval=80)
    else:
        reporter = None

    buffer = MetropolizedReplayBuffer(
        *pack_tensor_in_tuple(proposal.sample(500)),
        target_energy=target, proposal_energy=proposal,
        reporter=reporter
    )
    # test sampling
    assert len(buffer) == 500
    if use_product:
        s = buffer.sample(100)
        assert len(s) == 2
        assert s[0].shape == (100, 2)
        assert s[1].shape == (100, 2)
    else:
        assert buffer.sample(100).shape == (100, 2)
    # in the beginning, the samples do not follow the target distribution
    assert not target_prob_higher(*pack_tensor_in_tuple(buffer.sample(100)))

    # after updates the probability of buffer entries coming from the target should be higher
    acceptance_ratios = []
    for _ in range(100):
        n_accepted = buffer.update(*pack_tensor_in_tuple(proposal.sample(400)))
        acceptance_ratios.append(n_accepted/400)
    # in the end, acceptance should be rather infrequent
    assert np.mean(acceptance_ratios[:-10]) < 0.5
    # in the end, the buffer should be fairly converged
    assert target_prob_higher(*pack_tensor_in_tuple(buffer.sample(100)))


@pytest.mark.parametrize("use_product", [True, False])
def test_hdf5_file(tmpdir, ctx, use_product):
    pytest.importorskip("netCDF4")
    hdf5 = ReplayBufferHDF5File(tmpdir/"test.h5", "w")
    target = NormalDistribution(dim=2).to(**ctx)
    if use_product:
        target = ProductDistribution([target, target])
    samples = pack_tensor_in_tuple(target.sample(10))
    energies = target.energy(*samples)[:, 0]
    assert not hdf5.is_header_written
    hdf5.write_header(*samples)
    assert hdf5.is_header_written
    hdf5.write_buffer(*samples, energies=energies, step=0)
    hdf5.write_stats(energies, step=0, n_proposed=10, n_accepted=10)
    hdf5.write_accepted_samples(*samples, energies=energies, indices=np.arange(10), step=0, forced_update=True)
    hdf5.close()

    # as context mgr
    samples2 = pack_tensor_in_tuple(target.sample(20))
    energies2 = target.energy(*samples2)[:, 0]
    with ReplayBufferHDF5File(tmpdir/"test.h5", "a") as hdf5:
        assert len(hdf5) == 10
        assert hdf5.buffer_size == 10
        hdf5.write_stats(energies2, step=1, n_proposed=20, n_accepted=20)
        hdf5.write_accepted_samples(*samples2, energies=energies2, indices=np.arange(20), step=1, forced_update=True)
        hdf5.write_buffer(*samples2, energies=energies2, step=1)
        assert len(hdf5) == 30
        assert hdf5.buffer_size == 20

    # stats
    hdf5 = ReplayBufferHDF5File(tmpdir/"test.h5", "r")
    for key in hdf5.stats:
        assert hdf5.stats[key].shape == (2, )
    # buffer
    buffer_samples = hdf5.buffer["samples"]
    buffer_energies = hdf5.buffer["energies"]
    for s1, s2 in zip(buffer_samples, samples2):
        assert np.allclose(s2.detach().cpu().numpy().astype(float), s1.astype(float))
    assert np.allclose(buffer_energies.astype(float), energies2.detach().cpu().numpy().astype(float))

    # items
    assert np.allclose(hdf5[0]["energy"], as_numpy(energies[0]))
    assert np.allclose(hdf5[12:14]["energy"], as_numpy(energies2[2:4]))

    hdf5.close()


def test_hdf5_to_mdtraj(tmpdir, ctx):
    md = pytest.importorskip("mdtraj")
    bgmol = pytest.importorskip("bgmol")
    nframes = 2
    top = bgmol.systems.AlanineDipeptideTSF().mdtraj_topology
    samples = torch.randn(nframes, top.n_atoms, 3, **ctx)
    hdf5 = ReplayBufferHDF5File(tmpdir / "test.h5", "w")
    hdf5.write_header(samples)
    hdf5.write_accepted_samples(
        samples.reshape(nframes, -1),
        energies=torch.randn_like(samples[..., 0, 0]),
        indices=torch.arange(len(samples)),
        step=0,
        forced_update=True
    )
    traj = hdf5.as_mdtraj_trajectory(topology=top)
    assert traj.n_frames == nframes
    assert traj.xyz.shape == (nframes, top.n_atoms, 3)

