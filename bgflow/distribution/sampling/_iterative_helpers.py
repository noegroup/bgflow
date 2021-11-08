"""Helper classes and functions for iterative samplers."""

import torch


__all__ = ["AbstractSamplerState", "default_set_samples_hook", "default_extract_sample_hook"]


class AbstractSamplerState:
    """Defines the interface for implementations of the internal state of iterative samplers."""

    def as_dict(self):
        """Return a dictionary representing this instance. The dictionary has to define the
        keys that are used within `SamplerStep`s of an `IterativeSampler`, such as "samples", "energies", ...
        """
        raise NotImplementedError()

    def _replace(self, **kwargs):
        """Return a new object with changed fields.
        This function has to support all the keys that are used
        within `SamplerStep`s of an `IterativeSampler` as well as the keys "energies_up_to_date" and
        "forces_up_to_date"
        """
        raise NotImplementedError()

    def evaluate_energy_force(self, energy_model, evaluate_energies=True, evaluate_forces=True):
        """Return a new state with updated energies/forces."""
        state = self.as_dict()
        evaluate_energies = evaluate_energies and not state["energies_up_to_date"]
        energies = energy_model.energy(*state["samples"])[..., 0] if evaluate_energies else state["energies"]

        evaluate_forces = evaluate_forces and not state["forces_up_to_date"]
        forces = energy_model.force(*state["samples"]) if evaluate_forces else state["forces"]
        return self.replace(energies=energies, forces=forces)

    def replace(self, **kwargs):
        """Return a new state with updated fields."""

        # keep track of energies and forces
        state_dict = self.as_dict()
        if "energies" in kwargs:
            kwargs = {**kwargs, "energies_up_to_date": True}
        elif "samples" in kwargs:
            kwargs = {**kwargs, "energies_up_to_date": False}
        if "forces" in kwargs:
            kwargs = {**kwargs, "forces_up_to_date": True}
        elif "samples" in kwargs:
            kwargs = {**kwargs, "forces_up_to_date": False}

        # map to primary unit cell
        box_vectors = None
        if "box_vectors" in kwargs:
            box_vectors = kwargs["box_vectors"]
        elif "box_vectors" in state_dict:
            box_vectors = state_dict["box_vectors"]
        if "samples" in kwargs and box_vectors is not None:
            kwargs = {
                **kwargs,
                "samples": tuple(
                    _map_to_primary_cell(x, cell)
                    for x, cell in zip(kwargs["samples"], box_vectors)
                )
            }
        return self._replace(**kwargs)


def default_set_samples_hook(x):
    """by default, use samples as is"""
    return x


def default_extract_sample_hook(state: AbstractSamplerState):
    """Default extraction of samples from a SamplerState."""
    return state.as_dict()["samples"]


def _bmv(m, bv):
    """Batched matrix-vector multiply."""
    return torch.einsum("ij,...j->...i", m, bv)


def _map_to_primary_cell(x, cell):
    """Map coordinates to the primary unit cell of a periodic lattice.

    Parameters
    ----------
    x : torch.Tensor
        n-dimensional coordinates of shape (..., n), where n is the spatial dimension and ... denote an
        arbitrary number of batch dimensions.
    cell : torch.Tensor
        Lattice vectors (column-wise). Has to be upper triangular.
    """
    if cell is None:
        return x
    n = _bmv(torch.inverse(cell), x)
    n = torch.floor(n)
    return x - _bmv(cell, n)
