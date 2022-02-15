
__all__ = ["Energy"]


from typing import Union, Optional, Sequence
from collections.abc import Sequence as _Sequence
import warnings

import torch
import numpy as np
from ...utils.types import assert_numpy


def _is_non_empty_sequence_of_integers(x):
    return (
        isinstance(x, _Sequence) and (len(x) > 0) and all(isinstance(y, int) for y in x)
    )


def _is_sequence_of_non_empty_sequences_of_integers(x):
    return (
        isinstance(x, _Sequence)
        and len(x) > 0
        and all(_is_non_empty_sequence_of_integers(y) for y in x)
    )


def _parse_dim(dim):
    if isinstance(dim, int):
        return [torch.Size([dim])]
    if _is_non_empty_sequence_of_integers(dim):
        return [torch.Size(dim)]
    elif _is_sequence_of_non_empty_sequences_of_integers(dim):
        return list(map(torch.Size, dim))
    else:
        raise ValueError(
            f"dim must be either:"
            f"\n\t- an integer"
            f"\n\t- a non-empty list of integers"
            f"\n\t- a list with len > 1 containing non-empty lists containing integers"
        )


class Energy(torch.nn.Module):
    """
    Base class for all energy models.

    It supports energies defined over:
        - simple vector states of shape [..., D]
        - tensor states of shape [..., D1, D2, ..., Dn]
        - states composed of multiple tensors (x1, x2, x3, ...)
          where each xi is of form [..., D1, D2, ...., Dn]

    Each input can have multiple batch dimensions,
    so a final state could have shape
        ([B1, B2, ..., Bn, D1, D2, ..., Dn],
         ...,
         [B1, B2, ..., Bn, D'1, ..., D'1n]).

    which would return an energy tensor with shape
        ([B1, B2, ..., Bn, 1]).

    Forces are computed for each input by default.
    Here the convention is followed, that forces will have
    the same shape as the input state.

    To define the state shape, the parameter `dim` has to
    be of the following form:
        - an integer, e.g. d = 5
            then each event is a simple vector state
            of shape [..., 5]
        - a non-empty list of integers, e.g. d = [3, 6, 7]
            then each event is a tensor state of shape [..., 3, 6, 7]
        - a list of len > 1 containing non-empty integer lists,
            e.g. d = [[1, 3], [5, 3, 6]]. Then each event is
            a tuple of tensors of shape ([..., 1, 3], [..., 5, 3, 6])

    Parameters:
    -----------
    dim: Union[int, Sequence[int], Sequence[Sequence[int]]]
        The event shape of the states for which energies/forces ar computed.

    """

    def __init__(self, dim: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs):

        super().__init__(**kwargs)
        self._event_shapes = _parse_dim(dim)

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shapes[0]) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return int(torch.prod(torch.tensor(self.event_shape, dtype=int)))

    @property
    def event_shape(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore therefore there exists no single event shape."
                "Consider using Energy.event_shapes instead."
            )
        return self._event_shapes[0]

    @property
    def event_shapes(self):
        return self._event_shapes

    def _energy(self, *xs, **kwargs):
        raise NotImplementedError()

    def energy(self, *xs, temperature=1.0, **kwargs):
        assert len(xs) == len(
            self._event_shapes
        ), f"Expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        batch_shape = xs[0].shape[: -len(self._event_shapes[0])]
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert x.shape[: -len(s)] == batch_shape, (
                f"Inconsistent batch shapes."
                f"Input at index {i} has batch shape {x.shape[:-len(s)]}"
                f"however input at index 0 has batch shape {batch_shape}."
            )
            assert (
                x.shape[-len(s) :] == s
            ), f"Input at index {i} as wrong shape {x.shape[-len(s):]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    def force(
        self,
        *xs: Sequence[torch.Tensor],
        temperature: float = 1.0,
        ignore_indices: Optional[Sequence[int]] = None,
        no_grad: Union[bool, Sequence[int]] = False,
        **kwargs,
    ):
        """
        Computes forces with respect to the input tensors.

        If states are tuples of tensors, it returns a tuple of forces for each input tensor.
        If states are simple tensors / vectors it returns a single forces.

        Depending on the context it might be unnecessary to compute all input forces.
        For this case `ignore_indices` denotes those input tensors for which no forces.
        are to be computed.

        E.g. by setting `ignore_indices = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, None, fz)`.

        Furthermore, the forces will allow for taking high-order gradients by default.
        If this is unwanted, e.g. to save memory it can be turned off by setting `no_grad=True`.
        If higher-order gradients should be ignored for only a subset of inputs it can
        be specified by passing a list of ignore indices to `no_grad`.

        E.g. by setting `no_grad = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, fy, fz)`, where `fx` and `fz` allow for taking higher order gradients
        and `fy` will not.

        Parameters:
        -----------
        xs: *torch.Tensor
            Input tensor(s)
        temperature: float
            Temperature at which to compute forces
        ignore_indices: Sequence[int]
            Which inputs should be skipped in the force computation
        no_grad: Union[bool, Sequence[int]]
            Either specifies whether higher-order gradients should be computed at all,
            or specifies which inputs to leave out when computing higher-order gradients.
        """
        if ignore_indices is None:
            ignore_indices = []

        with torch.enable_grad():
            forces = []
            requires_grad_states = [x.requires_grad for x in xs]

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    x = x.requires_grad_(True)
                else:
                    x = x.requires_grad_(False)

            energy = self.energy(*xs, temperature=temperature, **kwargs)

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    if isinstance(no_grad, bool):
                        with_grad = not no_grad
                    else:
                        with_grad = i not in no_grad
                    force = -torch.autograd.grad(
                        energy.sum(), x, create_graph=with_grad,
                    )[0]
                    forces.append(force)
                    x.requires_grad_(requires_grad_states[i])
                else:
                    forces.append(None)

        forces = (*forces,)
        if len(self._event_shapes) == 1:
            forces = forces[0]
        return forces


class _BridgeEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bridge):
        energy, force, *_ = bridge.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None


_evaluate_bridge_energy = _BridgeEnergyWrapper.apply


class _Bridge:
    _FLOATING_TYPE = np.float64
    _SPATIAL_DIM = 3

    def __init__(self):
        self.last_energies = None
        self.last_forces = None

    def evaluate(
            self,
            positions: torch.Tensor,
            *args,
            evaluate_force: bool = True,
            evaluate_energy: bool = True,
            **kwargs
    ):
        shape = positions.shape
        assert shape[-2:] == (self.n_atoms, 3) or shape[-1] == self.n_atoms * 3
        energy_shape = shape[:-2] if shape[-2:] == (self.n_atoms, 3) else shape[:-1]
        # the stupid last dim
        energy_shape = [*energy_shape, 1]
        position_batch = assert_numpy(positions.reshape(-1, self.n_atoms, 3), arr_type=self._FLOATING_TYPE)

        energy_batch = np.zeros(energy_shape, dtype=position_batch.dtype)
        force_batch = np.zeros_like(position_batch)

        for i, pos in enumerate(position_batch):
            energy_batch[i], force_batch[i] = self._evaluate_single(
                pos,
                *args,
                evaluate_energy=evaluate_energy,
                evaluate_force=evaluate_force,
                **kwargs
            )

        energies = torch.tensor(energy_batch.reshape(*energy_shape)).to(positions)
        forces = torch.tensor(force_batch.reshape(*shape)).to(positions)

        # store
        self.last_energies = energies
        self.last_forces = forces

        return energies, forces

    def _evaluate_single(
            self,
            positions: torch.Tensor,
            *args,
            evaluate_force=True,
            evaluate_energy=True,
            **kwargs
    ):
        raise NotImplementedError

    @property
    def n_atoms(self):
        raise NotImplementedError()


class _BridgeEnergy(Energy):

    def __init__(self, bridge, two_event_dims=True):
        event_shape = (bridge.n_atoms, 3) if two_event_dims else (bridge.n_atoms * 3, )
        super().__init__(event_shape)
        self._bridge = bridge
        self._last_batch = None

    @property
    def last_batch(self):
        return self._last_batch

    @property
    def bridge(self):
        return self._bridge

    def _energy(self, batch, no_grads=False):
        # check if we have already computed this energy (hash of string representation should be sufficient)
        if hash(str(batch)) == self._last_batch:
            return self._bridge.last_energies
        else:
            self._last_batch = hash(str(batch))
            return _evaluate_bridge_energy(batch, self._bridge)

    def force(self, batch, temperature=None):
        # check if we have already computed this energy
        if hash(str(batch)) == self.last_batch:
            return self.bridge.last_forces
        else:
            self._last_batch = hash(str(batch))
            return self._bridge.evaluate(batch)[1]
