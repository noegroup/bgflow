
import torch
import numpy as np

from .energy.base import Energy
from .sampling.base import Sampler
from .distributions import CustomDistribution


__all__ = ["ProductEnergy", "ProductSampler", "ProductDistribution"]


class ProductEnergy(Energy):
    """Stack multiple energies together to form an energy on the product space.
    The energy on the product space is the sum of its independent components.

    Parameters
    ----------
    components : Sequence[Energy]
        The individual energies that form the direct product.
    cat_dim : int or None
        If None, the .energy function takes multiple tensors (one for each component).
        Otherwise, it expects one tensor that is then split along dimension `cat_dim`.

    Notes
    -----
    The underlying components have to be single-event energies.
    """
    def __init__(self, components, cat_dim=None, **kwargs):
        event_shapes, lengths = _stacked_event_shapes([c.event_shape for c in components], cat_dim)
        super().__init__(dim=event_shapes, **kwargs)
        self._components = torch.nn.ModuleList(components)
        self._cat_dim = cat_dim
        self._lengths = lengths

    def _energy(self, *xs):
        if self._cat_dim is None:
            assert len(xs) == len(self._components)
            energies = [dist.energy(x) for dist, x in zip(self._components, xs)]
        else:
            assert len(xs) == 1
            xs = xs[0].split(self._lengths, dim=self._cat_dim)
            energies = [dist.energy(x) for x, dist in zip(xs, self._components)]
        return torch.sum(torch.stack(energies, dim=-1), dim=-1)

    def __getitem__(self, index):
        return self._components[index]

    def __iter__(self):
        return self._components.__iter__()

    def __len__(self):
        return self._components.__len__()


class ProductSampler(Sampler):
    """Sampler on the product space.

    Parameters
    ----------
    components : Sequence[Sampler]
        The individual samplers that form the direct product.
    cat_dim : int or None
        If None, the .sample function generates multiple tensors (one for each component).
        Otherwise, it returns one tensor that is concatenated along dimension `cat_dim`.
    """
    def __init__(self, components, cat_dim=None, **kwargs):
        super().__init__(**kwargs)
        self._components = torch.nn.ModuleList(components)
        self._cat_dim = cat_dim

    def _sample(self, n_samples):
        samples = tuple(dist._sample(n_samples) for dist in self._components)
        if self._cat_dim is None:
            return samples
        else:
            return torch.cat(samples, dim=self._cat_dim)

    def _sample_with_temperature(self, n_samples, temperature=1.0):
        samples = tuple(dist._sample_with_temperature(n_samples, temperature) for dist in self._components)
        if self._cat_dim is None:
            return samples
        else:
            return torch.cat(samples, dim=self._cat_dim)

    def __getitem__(self, index):
        return self._components[index]

    def __iter__(self):
        return self._components.__iter__()

    def __len__(self):
        return self._components.__len__()


class ProductDistribution(CustomDistribution):
    """Distribution on a product space.
    Encapsulate multiple distributions in one object.

    Parameters
    ----------
    components : Iterable
        List of distributions.
    cat_dim : int or None
        The dimension along which samples from the individual components are concatenated.
        If None, don't concatenate.

    Notes
    -----
    The underlying components have to be single-event distributions.
    """

    def __init__(self, components, cat_dim=None):
        super().__init__(
            energy=ProductEnergy(components=components, cat_dim=cat_dim),
            sampler=ProductSampler(components=components, cat_dim=cat_dim)
        )


def _stacked_event_shapes(event_shapes, cat_dim):
    if cat_dim is None:
        return event_shapes, None
    else:
        lengths = [e[cat_dim] for e in event_shapes]
        shape = np.array(event_shapes[0])
        # assert that shapes are consistent
        for e in event_shapes:
            assert len(e) == len(shape)
            assert _shapes_consistent(e, shape, cat_dim)
        # concatenate events along dimensions `cat_dim`
        shape[cat_dim] = sum(s[cat_dim] for s in event_shapes)
        event_shapes = torch.Size(shape.tolist())
        return event_shapes, lengths


def _shapes_consistent(shape1, shape2, cat_dim):
    """check if shapes are the same in all dimensions but `cat_dim`"""
    diff = np.abs(np.array(shape1) - shape2)
    return diff.sum() == diff[cat_dim]