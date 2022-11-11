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
    Transformer
    AffineTransformer
    TruncatedGaussianTransformer
    ConditionalSplineTransformer
    ScalingLayer
    EntropyScalingLayer

===============================================================================
Continuous Normalizing Flows
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DiffEqFlow

Dynamics Functions
---------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    BlackBoxDynamics
    TimeIndependentDynamics
    KernelDynamics
    DensityDynamics


Jacobian Trace Estimators
------------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    BruteForceEstimator
    HutchinsonEstimator

===============================================================================
Stochastic Normalizing Flows
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MetropolisMCFlow
    BrownianFlow
    LangevinFlow
    StochasticAugmentation
    OpenMMStochasticFlow
    PathProbabilityIntegrator
    BrownianPathProbabilityIntegrator

===============================================================================
Internal Coordinate Transformations
===============================================================================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    RelativeInternalCoordinateTransformation
    GlobalInternalCoordinateTransformation
    MixedCoordinateTransformation
    WhitenFlow

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
    SequentialFlow

===============================================================================
Other
===============================================================================
Docs and/or classification required

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AffineFlow
    CheckerboardFlow
    BentIdentity
    FunnelFlow
    KroneckerProductFlow
    PseudoOrthogonalFlow
    InvertiblePPPP
    PPPPScheduler
    TorchTransform
    TriuFlow
    BNARFlow
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
from .modulo import *
