"""Tensor info objects and constants (dimension and topology) used in factories.
"""

import numpy as np
from ..nn.flow.crd_transform import GlobalInternalCoordinateTransformation
from typing import Sequence, Union
from collections import OrderedDict, namedtuple

__all__ = [
    "TensorInfo", "ShapeDictionary",
    "BONDS", "ANGLES", "TORSIONS", "FIXED",
    "ORIGIN", "ROTATION", "AUGMENTED",
    "TARGET"
]


class TensorInfo(
    namedtuple(
        "TensorInfo",
        ["name", "is_circular", "is_cartesian"],
        defaults=(False, False)
        # allow specifying bounds?
    )
):
    """Info about the tensor; its name and support

    Attributes
    ----------
    name : str
        The tensor's name.
    is_circular : bool
        Whether the domain that the tensor lives on has periodic boundary conditions.
    """
    pass


#: Default tensor info for bonds
BONDS = TensorInfo("BONDS", False, False)
#: Default tensor info for angles
ANGLES = TensorInfo("ANGLES", False, False)
#: Default tensor info for torsions
TORSIONS = TensorInfo("TORSIONS", True, False)
#: Default tensor info for fixed atoms in relative/mixed coordinate transforms
FIXED = TensorInfo("FIXED", False, True)
#: Default tensor info for global origin in global coordinate transform
ORIGIN = TensorInfo("ORIGIN", False, True)
#: Default tensor info for global rotation in global coordinate transform
ROTATION = TensorInfo("ROTATION", False, False)
#: Default tensor info for augmented dimensions
AUGMENTED = TensorInfo("AUGMENTED", False, False)
#: Default tensor info for the target space (e.g., Cartesian atom positions)
TARGET = TensorInfo("TARGET", False, True)


class ShapeDictionary(OrderedDict):
    """A dictionary of tensor shapes.
    Helper class to keep track of tensor shapes and periodicity when building a flow/Boltzmann generator.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_coordinate_transform(
            coordinate_transform,
            dim_augmented: int = 0,
            n_constraints: int = 0,
            remove_origin_and_rotation: bool = True
    ):
        """Static constructor. Create shape dictionary from a coordinate transform.

        Parameters
        ----------
        coordinate_transform : torch.nn.Module
            A coordinate transform, i.e. an instance of one of the classes in bgflow/nn/flow/crd_transform
        dim_augmented : int, optional
            For augmented normalizing flows; the number of augmented dimensions.
        n_constraints : int, optional
            Number of constrained bonds.

        Returns
        -------
        shape_info : ShapeDictionary
            The shape info defining the prior dimensions implied by the coordinate transform.
        """
        shape_info = ShapeDictionary()
        if coordinate_transform.dim_angles > 0:
            shape_info[BONDS] = (coordinate_transform.dim_bonds - n_constraints,)
        if coordinate_transform.dim_angles > 0:
            shape_info[ANGLES] = (coordinate_transform.dim_angles,)
        if coordinate_transform.dim_torsions > 0:
            shape_info[TORSIONS] = (coordinate_transform.dim_torsions,)
        if coordinate_transform.dim_fixed > 0:
            shape_info[FIXED] = (coordinate_transform.dim_fixed,)
        if dim_augmented > 0:
            shape_info[AUGMENTED] = (dim_augmented,)
        if isinstance(coordinate_transform, GlobalInternalCoordinateTransformation) and not remove_origin_and_rotation:
            shape_info[ORIGIN] = (1, 3)
            shape_info[ROTATION] = (3,)
        return shape_info

    def split(self, key: TensorInfo, into: Sequence[TensorInfo], sizes: Sequence[int], dim: int = -1) -> None:
        """Split one key in the dictionary into multiple keys.

        Parameters
        ----------
        key : TensorInfo
            A TensorInfo that is in the dictionary.
        into : Sequence[TensorInfo]
            TensorInfo instances that are inserted into the dictionary in place of the key.
        sizes : Sequence[int]
            Sizes of the inserted tensors along `dim`; has to have the same length as `into` and
            sum up to the size of the `key` tensor along `dim`
        dim : int, optional
            The dimension along which to split.
        """
        # remove one
        index = self.index(key)
        if sum(sizes) != self[key][dim]:
            raise ValueError(f"split sizes {sizes} do not sum up to total ({self[key]})")
        all_sizes = list(self[key])
        del self[key]
        # insert multiple
        for f in into:
            assert f not in self
        for el, size in zip(reversed(into), reversed(sizes)):
            all_sizes[dim] = size
            self.insert(el, index, tuple(all_sizes))

    def merge(self, keys: Sequence[TensorInfo], to: TensorInfo, index=None, dim: int = -1):
        """Concatenate multiple keys along a dimensions.

        Parameters
        ----------
        keys : Sequence[TensorInfo]
            The keys that are to be merged.
        to : TensorInfo
            The resulting key.
        index : Union[None, int], optional
            The index in the (ordered) dictionary at which to insert the new tensor.
        dim : int, optional
            The dimension along which to concatenate.
        """
        # remove multiple
        size = sum(self[f][dim] for f in keys)
        all_sizes = list(self[keys[0]])
        # TODO: check that other dimensions are compatible
        all_sizes[dim] = size
        first_index = min(self.index(f) for f in keys)
        for f in keys:
            del self[f]
        # insert one
        assert to not in self
        if index is None:
            index = first_index
        self.insert(to, index, tuple(all_sizes))

    def replace(self, key: TensorInfo, other: Union[str, TensorInfo]):
        """Rename/Replace a key.

        Parameters
        ----------
        key : TensorInfo
            The key to be replaced.
        other :  Union[str,TensorInfo]
            The key (or a name) that replaces the old key.
        """
        if isinstance(other, str):
            other = key._replace(name=other)
        self.insert(other, self.index(key), self[key])
        del self[key]
        return other

    def copy(self):
        """Copy this dictionary."""
        clone = ShapeDictionary()
        for key in self:
            clone[key] = self[key]
        return clone

    def insert(self, key, index, size):
        """Insert a key at a certain index.

        Parameters
        ----------
        key : TensorInfo
            The key to insert.
        index : int
            The index at which to insert the key into the dictionary.
        size : Sequence[int]
            The value; i.e., the tensor shape associated with the key.
        """
        if index < 0:
            index = len(self) - index
        assert key not in self
        self[key] = size  # append
        for i, key in enumerate(list(self)):
            if index <= i < len(self) - 1:
                self.move_to_end(key)

    def index(self, key: TensorInfo, keys: Union[None, Sequence[TensorInfo]] = None):
        """The index (position) of a key in the dictionary.

        Parameters
        ----------
        key : TensorInfo
            The key that is searched.
        keys : Sequence[TensorInfo]], optional
            The keys in which to search; if None, search the entire dictionary (self)

        Returns
        -------
        index : int
            The position.
        """
        keys = self if keys is None else keys
        return list(keys).index(key)

    def names(self, keys: Union[None, Sequence[TensorInfo]] = None):
        """The names of all keys.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary (self)

        Returns
        -------
        names : list[str]
            all names
        """
        keys = self if keys is None else keys
        return [key.name for key in keys]

    def dim_all(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """The total dimension of all tensors along a given axis `dim`.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        dim_all : int
            Total length.
        """
        keys = self if keys is None else keys
        return sum(self[key][dim] for key in keys)

    def dim_circular(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """The total dimension of all circular tensors along a given axis `dim`.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        dim_circular : int
            Total number of circular elements.
        """
        keys = self if keys is None else keys
        return sum(self[key][dim] for key in keys if key.is_circular)

    def dim_noncircular(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """see `dim_circular`"""
        keys = self if keys is None else keys
        return sum(self[key][dim] for key in keys if not key.is_circular)

    def is_circular(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """The total dimension of all circular tensors along a given axis `dim`.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        is_circular : np.array
            A boolean array that contains `True` for each circular indices and `False` for others
        """
        keys = self if keys is None else keys
        return np.concatenate([np.ones(self[key][dim]) * key.is_circular for key in keys]).astype(bool)

    def circular_indices(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """Indices that are circular.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        circular_indices : np.array
            An array of data type int that contains the indices of all circular elements as seen in
            the context of all keys concatenated.
        """
        keys = self if keys is None else keys
        return np.arange(self.dim_all(keys, dim))[self.is_circular(keys, dim)]

    def dim_cartesian(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """The total dimension of all circular tensors along a given axis `dim`.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        dim_circular : int
            Total number of circular elements.
        """
        keys = self if keys is None else keys
        return sum(self[key][dim] for key in keys if key.is_cartesian)

    def dim_noncartesian(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """see `dim_circular`"""
        keys = self if keys is None else keys
        return sum(self[key][dim] for key in keys if not key.is_cartesian)

    def is_cartesian(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """The total dimension of all circular tensors along a given axis `dim`.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        is_circular : np.array
            A boolean array that contains `True` for each circular indices and `False` for others
        """
        keys = self if keys is None else keys
        return np.concatenate([np.ones(self[key][dim]) * key.is_cartesian for key in keys]).astype(bool)

    def cartesian_indices(self, keys: Union[None, Sequence[TensorInfo]] = None, dim: int = -1):
        """Indices that are circular.

        Parameters
        ----------
        keys : Sequence[TensorInfo]], optional
            The keys over which to iterate; if None, iterate over the entire dictionary.
        dim : int, optional
            The axis over which to sum.

        Returns
        -------
        circular_indices : np.array
            An array of data type int that contains the indices of all circular elements as seen in
            the context of all keys concatenated.
        """
        keys = self if keys is None else keys
        return np.arange(self.dim_all(keys, dim))[self.is_cartesian(keys, dim)]
