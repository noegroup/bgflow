import pytest
import torch
from bgflow.distribution import NormalDistribution
from bgflow.nn.flow.dynamics import TimeIndependentDynamics
from bgflow.nn.flow.estimator import HutchinsonEstimator
from bgflow.nn import DenseNet
from bgflow.utils import brute_force_jacobian_trace


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("rademacher", [True, False])
def test_hutchinson_estimator(dim, rademacher):
    # Test trace estimation of the hutchinson estimator for small dimensions, where it is less noisy
    n_batch = 1024
    time_independent_dynamics = TimeIndependentDynamics(
        DenseNet([dim, 16, 16, dim], activation=torch.nn.Tanh()))
    hutchinson_estimator = HutchinsonEstimator(rademacher)
    normal_distribution = NormalDistribution(dim)
    x = normal_distribution.sample(n_batch)
    y, trace = hutchinson_estimator(time_independent_dynamics, None, x)
    brute_force_trace = brute_force_jacobian_trace(y, x)
    if rademacher and dim == 1:
        # Hutchinson is exact for rademacher noise and dim=1
        assert torch.allclose(trace.mean(), -brute_force_trace.mean(), atol=1e-6)
    else:
        assert torch.allclose(trace.mean(), -brute_force_trace.mean(), atol=1e-1)


@pytest.mark.parametrize("rademacher", [True, False])
def test_test_hutchinson_estimator_reset_noise(rademacher):
    # Test if the noise vector is resetted to deal with different shape
    dim = 10
    time_independent_dynamics = TimeIndependentDynamics(
        DenseNet([dim, 16, 16, dim], activation=torch.nn.Tanh()))
    hutchinson_estimator = HutchinsonEstimator(rademacher)
    normal_distribution = NormalDistribution(dim)

    x = normal_distribution.sample(100)
    _, _ = hutchinson_estimator(time_independent_dynamics, None, x)
    x = normal_distribution.sample(10)
    hutchinson_estimator.reset_noise()
    # this will fail if the noise is not resetted
    _, _ = hutchinson_estimator(time_independent_dynamics, None, x)
