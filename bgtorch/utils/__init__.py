from .train import IndexBatchIterator 
from .shape import tile
from .types import (
    is_list_or_tuple,
    assert_numpy
)
from .autograd import (
    brute_force_jacobian, 
    brute_force_jacobian_trace 
)
from .geometry import (
    distance_vectors,
    distances_from_vectors
)
from .rbf_kernels import (
    kernelize_with_rbf,
    compute_gammas,
    RbfEncoder,
    rbf_kernels
)