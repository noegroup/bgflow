
import os
import gc
import numpy as np
import torch
from ...utils.types import unpack_tensor_tuple
from .dataset import DataSetSampler
from .mcmc import metropolis_accept


__all__ = ["MetropolizedReplayBuffer", "ReplayBufferHDF5File", "ReplayBufferHDF5Reporter"]


class MetropolizedReplayBuffer(DataSetSampler):
    """A set of samples that can be updated with Monte Carlo moves.

    Parameters
    ----------
    *data : Sequence[torch.Tensor]
        The samples that are in the buffer upon initialization.
    target_energy : bg.Energy, optional
        The target energy that is used to evaluate the Metropolis criterion.
    proposal_energy : bg.Energy, optional
        The energy of the (non-conditional) proposal distribution.
    energies : torch.Tensor, optional
        The target energies corresponding to the initial samples.
    temperature_ratio : float, optional
        A scaling factor to the target temperature.
    reporter : ReplayBufferReporter, optional
        A reporter instance to write trajectories and statistics to output files.
    """

    def __init__(
            self,
            *data,
            target_energy=None,
            proposal_energy=None,
            energies=None,
            temperature_scaling=1.0,
            reporter=None,
    ):
        if energies is None:
            if target_energy is None:
                raise ValueError("Either target_energy or energies has to specified.")
            energies = target_energy.energy(*data)[:,0]
        else:
            n_event_dims_0 = len(data[0].shape) - len(target_energy.event_shapes[0])
            expected_energies_shape = data[0].shape[:-n_event_dims_0]
            if energies.shape != expected_energies_shape:
                raise ValueError(f"Expected energy shape {expected_energies_shape} to be consistent "
                                 f"with the data shape {data[0].shape} but got shape{energies.shape}.")
        super().__init__(*data, energies)
        self._target_energy = target_energy
        self._proposal_energy = proposal_energy
        self.reporter = reporter
        if reporter is not None:
            reporter.write_buffer(*data, energies=energies)
        self.temperature_scaling = temperature_scaling

    def _sample(self, n_samples, *args, **kwargs):
        *samples, _ = super()._sample(n_samples, *args, **kwargs)
        return unpack_tensor_tuple(samples)

    def update(self, *proposals, energies=None, proposal_energies=None, forced_update=False):
        # compute energies
        if energies is None:
            energies = self._target_energy.energy(*proposals)[:, 0]
        # compute proposal energies
        if proposal_energies is None:
            proposal_energies = self._proposal_energy.energy(*proposals)[:, 0]
        # select random elements from the replay buffer
        rand_indices = torch.randperm(len(self))[:len(proposals[0])]
        *rand_samples, rand_energies = [data[rand_indices].to(proposals[0]) for data in self.data]
        # Metropolis criterion
        accepted = metropolis_accept(  # SHAPES!
            current_energies=rand_energies,
            proposed_energies=energies,
            proposal_delta_log_prob=-proposal_energies + self._proposal_energy.energy(*rand_samples)[:, 0]
            # log g(x'|x) - log g(x|x') = log g(x') - log g(x) = -u(x') + u(x)
        )
        if forced_update:
            accepted[:] = True
        # replace samples and energies in buffer
        n_accepted = accepted.sum().item()
        accepted_indices = rand_indices[accepted]
        accepted_samples = [proposal[accepted].to(self.data[0]) for proposal in proposals]
        accepted_energies = [energies[accepted].to(self.data[0])]
        for i, accepted_item in enumerate(accepted_samples + accepted_energies):
            self.data[i][accepted_indices, ...] = accepted_item
        if self.reporter is not None:
            self.reporter.write(
                *accepted_samples,
                buffer=self,
                energies=accepted_energies[0],
                indices=accepted_indices,
                forced_update=forced_update,
                n_proposed=len(energies)
            )
        return n_accepted

    @property
    def energies(self):
        return self.data[-1]

    @property
    def samples(self):
        return (*self.data[:-1],)


class ReplayBufferHDF5Reporter:
    """A reporter that can be used to write the output of MetropolizedReplayBuffer sampling
    to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output filename ending on ".h5"
    mode : str, optional
        The opening mode ("r" for read, "r+" for "read-write", etc.).
        By default, the file will be opened in "r+" mode if it exists and in "w" mode if not.
    write_buffer_interval : int, optional
        After each i-th step, the whole replay buffer is written to the file.
    """
    def __init__(self, filename, mode=None, write_buffer_interval=100):
        if mode is None:
            mode = "r+" if os.path.isfile(filename) else "w"
        self.file = ReplayBufferHDF5File(filename, mode)
        if self.file.is_header_written:
            self.step = self.file.stats_size
        else:
            self.step = 0
        self.write_buffer_interval = write_buffer_interval

    def write_buffer(self, *samples, energies):
        """Write the content of the replay buffer.

        Parameters
        ----------
        *samples : torch.Tensor
        energies : torch.Tensor
        """
        self.file.write_buffer(*samples, energies=energies, step=self.step)

    def _write_accepted_samples(self, *samples, energies, indices, forced_update):
        self.file.write_accepted_samples(
            *samples,
            energies=energies,
            indices=indices,
            step=self.step,
            forced_update=forced_update
        )

    def _write_stats(self, energies, n_proposed, n_accepted):
        self.file.write_stats(energies, step=self.step, n_proposed=n_proposed, n_accepted=n_accepted)

    def write(self, *samples, buffer, energies, indices, forced_update, n_proposed):
        """Write one step.

        Parameters
        ----------
        *samples : torch.Tensor
            Accepted samples.
        buffer : MetropolizedReplayBuffer
            The buffer.
        energies : torch.Tensor
            Energies of accepted samples.
        indices : np.ndarray
            Buffer indices, which will be replaced by the accepted samples.
        forced_update : bool
            Whether this was a forced buffer update.
        n_proposed : int
            Number of samples that were proposed to the buffer in this step.
        """
        self._write_accepted_samples(*samples, energies=energies, indices=indices, forced_update=forced_update)
        self._write_stats(buffer.energies, n_proposed, n_accepted=len(energies))
        if self.step % self.write_buffer_interval == 0:
            self.write_buffer(*buffer.samples, energies=buffer.energies)
        self.step += 1


class ReplayBufferHDF5File:
    """HDF5 file storing data from the replay buffer.

    Parameters
    ----------
    filename : str
        The filename of this HDF5 file.
    mode : str
        The opening mode ("r" for read, "r+" for "read-write", etc.)

    Attributes
    ----------
    buffer : dict[str, np.ndarray]
        A dictionary {"samples": list[np.ndarray], "energy": np.ndarray}
    is_header_written : bool
        Whether the file header including variable definitions etc. is already written.
    stats : dict[str, np.ndarray]
        A dictionary of stats.

    """
    def __init__(self, filename, mode):
        import netCDF4 as nc
        self.filename = filename
        self.mode = mode
        self.dataset = nc.Dataset(self.filename, self.mode)

    def write_header(self, *samples):
        """Write the file header that contains group, variable, and dimension definitions."""
        self.dataset.createDimension("steps", None)
        self.dataset.createDimension("acceptances", None)
        self.dataset.createDimension("buffer_size", None)
        # stats group
        stats = self.dataset.createGroup("stats")
        stats.createVariable("step", "u8", ("steps",))
        stats.createVariable("mean_energy", "f4", ("steps",))
        stats.createVariable("min_energy", "f4", ("steps",))
        stats.createVariable("max_energy", "f4", ("steps",))
        stats.createVariable("median_energy", "f4", ("steps",))
        stats.createVariable("n_proposed", "u4", ("steps",))
        stats.createVariable("n_accepted", "u4", ("steps",))
        stats.createVariable("buffer_size", "u4", ("steps",))
        # groups containing accepted samples and the buffer state
        data = self.dataset.createGroup("data")
        buffer = self.dataset.createGroup("buffer")
        for i, s in enumerate(samples):
            sample_dims = ["acceptances", ]
            buffer_dims = ["buffer_size", ]
            for j, d in enumerate(s.shape[1:]):
                name = f"sampledim_{i}_{j}"
                self.dataset.createDimension(name, d)
                sample_dims.append(name)
                buffer_dims.append(name)
            data.createVariable(f"sample{i}", "f4", sample_dims)
            buffer.createVariable(f"sample{i}", "f4", buffer_dims)
        data.createVariable("energy", "f4", ("acceptances",))
        data.createVariable("running_index", "u8", ("acceptances",))
        data.createVariable("step", "u4", ("acceptances",))
        data.createVariable("buffer_index", "u4", ("acceptances",))
        data.createVariable("forced_update", "b", ("acceptances",))
        data.createVariable("last_buffer_write", "u8", ("acceptances",))
        buffer.createVariable("energy", "f4", ("buffer_size",))
        buffer.createVariable("step", "u8")

    def _append_samples(self, *samples, energies, buffer_indices, step, forced_update, last_buffer_write):
        """Append samples to the data group."""
        pos = slice(len(self), len(self)+len(energies))
        data_group = self.dataset["data"]
        for var, sampl in zip(self._sample_fields(), samples):
            data_group[var][pos] = sampl.detach().cpu().numpy()
        data_group["energy"][pos] = energies.detach().cpu().numpy()
        data_group["buffer_index"][pos] = buffer_indices
        data_group["step"][pos] = step
        data_group["forced_update"][pos] = forced_update
        data_group["last_buffer_write"][pos] = last_buffer_write
        data_group["running_index"][pos] = np.arange(len(self), len(self)+len(energies))

    def write_accepted_samples(self, *samples, energies, indices, step, forced_update):
        """Write samples to the HDF5 file.

        Parameters
        ----------
        *samples : torch.Tensor
            Accepted samples.
        energies : torch.Tensor
            Energies of the accepted samples.
        indices : np.ndarray
            Indices in the buffer that are replaced by the samples.
        step : int
            Running index of this update step.
        forced_update : bool
            Whether the samples were added by a forced update.
        """
        if not self.is_header_written:
            self.write_header(*samples)
        n_samples = len(samples[0])
        assert all(len(s) == n_samples for s in samples)
        assert len(energies) == n_samples
        assert len(indices) == n_samples
        self._append_samples(
            *samples,
            energies=energies,
            buffer_indices=indices,
            step=step,
            forced_update=forced_update,
            last_buffer_write=self.dataset["buffer"]["step"][:],
        )

    def write_stats(self, energies, step, n_proposed, n_accepted):
        """Write statistics to file.

        Parameters
        ----------
        energies : torch.Tensor
        step : int
        n_proposed : int
        n_accepted : int
        """
        if not self.is_header_written:
            raise AttributeError("You have to write the header first.")
        pos = self.stats_size
        stat_group = self.dataset["stats"]
        stat_group["min_energy"][pos] = energies.min().item()
        stat_group["mean_energy"][pos] = energies.mean().item()
        stat_group["max_energy"][pos] = energies.max().item()
        stat_group["median_energy"][pos] = energies.median().item()
        stat_group["buffer_size"][pos] = len(energies)
        stat_group["n_proposed"][pos] = n_proposed
        stat_group["n_accepted"][pos] = n_accepted
        stat_group["step"][pos] = step

    def write_buffer(self, *samples, energies, step):
        """Write buffer to file.

        Parameters
        ----------
        *samples : torch.Tensor
            tensors of samples

        energies : torch.Tensor
            tensor of energies

        Notes
        -----
        This overwrites the current buffer.
        """
        if not self.is_header_written:
            self.write_header(*samples)
        buffer_group = self.dataset["buffer"]
        for var, sample in zip(self._sample_fields(), samples):
            buffer_group[var][:] = sample.detach().cpu().numpy()
        buffer_group["energy"][:] = energies.detach().cpu().numpy()
        buffer_group["step"][:] = step

    def _sample_fields(self):
        for var in self.dataset["buffer"].variables:
            if "sample" in var:
                yield var

    @property
    def buffer(self):
        return {
            "samples": [np.array(self.dataset["buffer"][var]) for var in self._sample_fields()],
            "energies": np.array(self.dataset["buffer"]["energy"])
        }

    @property
    def stats(self):
        return {stat: np.array(self.dataset["stats"][stat][:]) for stat in self.dataset["stats"].variables}

    @property
    def is_header_written(self):
        return "buffer" in self.dataset.groups and "energy" in self.dataset["buffer"].variables

    def __getitem__(self, indices):
        data_group = self.dataset["data"]
        result_dict = {
            "samples": [np.array(data_group[var][indices]) for var in self._sample_fields()]
        }
        for var in data_group.variables:
            if not "sample" in var:
                result_dict[var] = np.array(data_group[var][indices])
        return result_dict

    def as_mdtraj_trajectory(self, topology, indices=slice(None)):
        import mdtraj as md
        assert len(list(self._sample_fields())) == 1
        data = self[indices]
        return md.Trajectory(
            xyz=data["samples"][0],
            topology=topology
        )

    def __len__(self):
        return self.dataset.dimensions["acceptances"].size

    @property
    def buffer_size(self):
        return self.dataset.dimensions["buffer_size"].size

    @property
    def stats_size(self):
        return self.dataset.dimensions["steps"].size

    def close(self):
        """Close the file."""
        self.dataset.close()

    @staticmethod
    def close_all_h5():
        """Close all netCDF4 datasets in the global scope."""
        import netCDF4 as nc
        for obj in gc.get_objects():  # Browse through ALL objects
            if isinstance(obj, nc.Dataset):  # Just HDF5 files
                try:
                    obj.close()
                except RuntimeError:
                    pass  # Was already closed

    def __enter__(self, *args, **kwargs):
        self.dataset.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        return self.dataset.__exit__(*args, **kwargs)
