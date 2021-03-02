from collections.abc import Sequence
import warnings

import torch

__all__ = ["Energy"]


def _is_non_empty_sequence_of_integers(x):
    return (
        isinstance(x, Sequence) and (len(x) > 0) and all(isinstance(y, int) for y in x)
    )


def _is_sequence_of_non_empty_sequences_of_integers(x):
    return (
        isinstance(x, Sequence)
        and len(x) > 1
        and all(isinstance(y, Sequence) and (len(y)) > 0 for y in x)
        and all(isinstance(z, int) for y in x for z in y)
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
    def __init__(self, dim):
        super().__init__()
        self._event_shapes = _parse_dim(dim)

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shape) > 0:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return torch.prod(torch.tensor(self.event_shape, dtype=int))

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
        ), f"expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert (
                len(x.shape) > 0 and x.shape[1:] == s
            ), f"input at index {i} as wrong shape {x.shape[1:]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    def force(
        self, *xs, temperature=1.0, with_grad=True, ignore_indices=None, **kwargs
    ):
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
                    force = -torch.autograd.grad(
                        energy.sum(), x, create_graph=with_grad,
                    )[0]
                    forces.append(force)
                    x.requires_grad_(requires_grad_states[i])
                else:
                    forces.append(None)

        if len(self._event_shapes) == 1:
            forces = forces[0]
        return forces
