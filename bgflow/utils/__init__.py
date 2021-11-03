"""
.. currentmodule: bgflow.utils

===============================================================================
Distance utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    distance_vectors
    distances_from_vectors
    remove_mean
    compute_distances


"""

from .train import IndexBatchIterator, LossReporter
from .shape import tile
from .types import *
from .autograd import *

from .geometry import (
    distance_vectors,
    distances_from_vectors,
    remove_mean,
    compute_distances
)
from .rbf_kernels import (
    kernelize_with_rbf,
    compute_gammas,
    RbfEncoder,
    rbf_kernels
)
