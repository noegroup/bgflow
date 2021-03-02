import pytest
import torch
from bgtorch.distribution import NormalDistribution
from bgtorch.nn.flow import DiffEqFlow
from bgtorch.nn.flow.dynamics import KernelDynamics
from bgtorch.utils import brute_force_jacobian_trace


@pytest.mark.parametrize("n_particles", [2, 3])
@pytest.mark.parametrize("n_dimensions", [2, 3])
def test_kernel_dynamics(n_particles, n_dimensions):
    # Build flow with kernel dynamics and run initial config.
    dim = n_particles * n_dimensions
    n_samples = 100
    prior = NormalDistribution(dim)
    latent = prior.sample(n_samples)

    d_max = 8
    mus = torch.linspace(0, d_max, 10)
    gammas = 0.3 * torch.ones(len(mus))

    mus_time = torch.linspace(0, 1, 5)
    gammas_time = 0.3 * torch.ones(len(mus_time))

    kernel_dynamics = KernelDynamics(n_particles, n_dimensions, mus, gammas, optimize_d_gammas=True,
                                     optimize_t_gammas=True, mus_time=mus_time, gammas_time=gammas_time)

    flow = DiffEqFlow(
        dynamics=kernel_dynamics
    )
    try:
        samples, dlogp = flow(latent)
    except ImportError:
        pytest.skip("Test requires torchdiffeq.")

    assert samples.shape == torch.Size([n_samples, dim])
    assert dlogp.shape == torch.Size([n_samples, 1])

    flow._use_checkpoints = True
    options = {
        "Nt": 20,
        "method": "RK4"
    }
    flow._kwargs = options

    try:
        samples, dlogp = flow(latent)
    except ImportError:
        pytest.skip("Test requires anode.")

    assert samples.shape == torch.Size([n_samples, dim])
    assert dlogp.shape == torch.Size([n_samples, 1])


@pytest.mark.parametrize("n_particles", [2, 3])
@pytest.mark.parametrize("n_dimensions", [2, 3])
def test_kernel_dynamics_trace(n_particles, n_dimensions):
    # Test if the trace computation of the kernel dynamics is correct.

    d_max = 8
    mus = torch.linspace(0, d_max, 10)
    gammas = 0.3 * torch.ones(len(mus))

    mus_time = torch.linspace(0, 1, 5)
    gammas_time = 0.3 * torch.ones(len(mus_time))

    kernel_dynamics = KernelDynamics(n_particles, n_dimensions, mus, gammas, mus_time=mus_time, gammas_time=gammas_time)
    x = torch.Tensor(1, n_particles * n_dimensions).normal_().requires_grad_(True)
    y, trace = kernel_dynamics(1., x)
    brute_force_trace = brute_force_jacobian_trace(y, x)

    # The kernel dynamics outputs the negative trace
    assert torch.allclose(trace.sum(), -brute_force_trace[0], atol=1e-4)
