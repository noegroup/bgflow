from .train import IndexBatchIterator 
from .shape import tile
from .types import *
from .autograd import *

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