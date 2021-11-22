"""
.. currentmodule: bgflow.nn.flow

===============================================================================
Coupling flows
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CouplingFlow
    SplitFlow
    MergeFlow
    SwapFlow
    WrapFlow
    AffineFlow


===============================================================================
Internal Coordinate Transformation
===============================================================================

===============================================================================
Continuous Normalizing Flows
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DiffEqFlow
    BlackBoxDynamics


===============================================================================
Stochastic Normalizing Flows
===============================================================================

===============================================================================
Autoregressive Flows
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    BNARFlow


===============================================================================
CDF Transformations
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CDFTransform
    DistributionTransferFlow
    ConstrainGaussianFlow

===============================================================================
Base
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Flow
    InverseFlow

===============================================================================
Other
===============================================================================
Docs and/or classification required

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CheckerboardFlow
    BentIdentity
    FunnelFlow
    KroneckerProductFlow
    PseudoOrthogonalFlow
"""

from .base import *
from .crd_transform import *
from .dynamics import *
from .estimator import *
from .stochastic import *
from .transformer import *

from .affine import *
from .coupling import *
from .funnel import FunnelFlow
from .spline import LinearSplineFlow
from .kronecker import KroneckerProductFlow
from .sequential import SequentialFlow
from .inverted import *
from .checkerboard import CheckerboardFlow
from .bnaf import BNARFlow
from .elementwise import *
from .orthogonal import *
from .triangular import *
from .pppp import *
from .diffeq import DiffEqFlow
from .cdf import *
from .torchtransform import *
