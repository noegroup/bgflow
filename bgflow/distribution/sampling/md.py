"""Molecular dynamics sampling steps."""

import torch

from .iterative import SamplerStep, SamplerState
from ..energy.openmm import OpenMMBridge
from ...utils.types import pack_tensor_in_list


__all__ = ["OpenMMStep"]


class OpenMMStep(SamplerStep):
    """Take simulation steps in OpenMM.

    Parameters
    ----------
    openmm_bridge : OpenMMBridge
        The bridge instance that defines the system, integrator, platform, and number of simulation steps.
    """
    def __init__(self, openmm_bridge: OpenMMBridge):
        super().__init__(n_steps=1)
        # n_steps is steered internally through the n_simulation_steps parameter of the OpenMMBridge
        self.bridge = openmm_bridge

    def _step(self, state: SamplerState):
        if len(state.samples) != 1:
            raise ValueError("OpenMMStep does not support multiple event tensors.")
        if state.box_vectors is not None:
            assert torch.allclose(
                torch.tril(state.box_vectors, diagonal=-1),
                torch.zeros_like(state.box_vectors)
            ), "OpenMM can only process box vectors with zeros under the diagonal."
        openmm_output = self.bridge.evaluate(
            batch=state.samples[0],
            box_vector_batch=(
                None if state.box_vectors is None
                else state.box_vectors
            ),
            velocity_batch=None if state.velocities is None else state.velocities[0],
            evaluate_force=True,
            evaluate_energy=True,
            evaluate_positions=True,
            evaluate_velocities=True,
            evaluate_box_vectors=False if state.box_vectors is None else True,
            evaluate_path_probability_ratio=False
        )
        state.samples = pack_tensor_in_list(openmm_output.new_positions)
        state.box_vectors = (
            None if openmm_output.new_box_vectors is None
            else openmm_output.new_box_vectors
        )
        state.velocities = pack_tensor_in_list(openmm_output.new_velocities)
        state.forces = pack_tensor_in_list(openmm_output.forces)
        state.energies = pack_tensor_in_list(openmm_output.energies)
        return state
