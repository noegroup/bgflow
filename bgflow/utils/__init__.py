"""
.. currentmodule: bgflow.utils

===============================================================================
Geometry Utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    distance_vectors
    distances_from_vectors
    remove_mean
    compute_distances
    compute_gammas
    kernelize_with_rbf
    RbfEncoder
    rbf_kernels

===============================================================================
Jacobian Computation
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    brute_force_jacobian_trace
    brute_force_jacobian
    "batch_jacobian
    get_jacobian
    requires_grad

===============================================================================
Types
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    is_list_or_tuple
    assert_numpy
    as_numpy

===============================================================================
Free Energy Estimation
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bennett_acceptance_ratio

===============================================================================
Training utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    IndexBatchIterator
    LossReporter


===============================================================================
Training utilities
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ClipGradient

"""

from .train import IndexBatchIterator, LossReporter, ClipGradient
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

from .free_energy import *
