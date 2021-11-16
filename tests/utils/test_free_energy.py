import warnings

import numpy as np
import pytest
import torch
from bgflow.distribution import NormalDistribution

from bgflow.utils.free_energy import bennett_acceptance_ratio


@pytest.mark.parametrize("method", ["torch", "pymbar"])
@pytest.mark.parametrize("compute_uncertainty", [True, False])
def test_bar(ctx, method, compute_uncertainty):
    pytest.importorskip("pymbar")
    dim = 1
    energy1 = NormalDistribution(dim, mean=torch.zeros(dim, **ctx))
    energy2 = NormalDistribution(dim, mean=0.2*torch.ones(dim, **ctx))
    samples1 = energy1.sample(10000)
    samples2 = energy2.sample(20000)

    free_energy, uncertainty = bennett_acceptance_ratio(
        forward_work=(1.0 + energy2.energy(samples1)) - energy1.energy(samples1),
        reverse_work=energy1.energy(samples2) - (1.0 + energy2.energy(samples2)),
        implementation=method,
        compute_uncertainty=compute_uncertainty
    )
    assert free_energy.item() == pytest.approx(1., abs=1e-2)
    if compute_uncertainty:
        assert uncertainty.item() < 1e-2
    else:
        assert uncertainty is None


@pytest.mark.parametrize("method", ["torch", "pymbar"])
@pytest.mark.parametrize("warn", [True, False])
def test_bar_no_convergence(ctx, method, warn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pytest.importorskip("pymbar")
        dim = 1
        energy1 = NormalDistribution(dim, mean=-1e20*torch.ones(dim, **ctx))
        energy2 = NormalDistribution(dim, mean=1e20*torch.ones(dim, **ctx))
        samples1 = energy1.sample(5)
        samples2 = energy2.sample(5)

        if warn:
            # test if warning is raised
            with pytest.warns(UserWarning, match="BAR could not"):
                free_energy, uncertainty = bennett_acceptance_ratio(
                    forward_work=(1.0 + energy2.energy(samples1)) - energy1.energy(samples1),
                    reverse_work=energy1.energy(samples2) - (1.0 + energy2.energy(samples2)),
                    implementation=method,
                    warn=warn
                )
        else:
            free_energy, uncertainty = bennett_acceptance_ratio(
                forward_work=(1.0 + energy2.energy(samples1)) - energy1.energy(samples1),
                reverse_work=energy1.energy(samples2) - (1.0 + energy2.energy(samples2)),
                implementation=method,
                warn=warn
            )
        assert np.isnan(free_energy.item())
        assert np.isnan(uncertainty.item())


def test_bar_uncertainty(ctx):
    """test consistency with the reference implementation"""
    pytest.importorskip("pymbar")
    dim = 1
    energy1 = NormalDistribution(dim, mean=torch.zeros(dim, **ctx))
    energy2 = NormalDistribution(dim, mean=0.2*torch.ones(dim, **ctx))  # will be multiplied by e
    samples1 = energy1.sample(1000)
    samples2 = energy2.sample(2000)

    free_energy1, uncertainty1 = bennett_acceptance_ratio(
        forward_work=(1.0 + energy2.energy(samples1)) - energy1.energy(samples1),
        reverse_work=energy1.energy(samples2) - (1.0 + energy2.energy(samples2)),
        implementation="torch"
    )
    free_energy2, uncertainty2 = bennett_acceptance_ratio(
        forward_work=(1.0 + energy2.energy(samples1)) - energy1.energy(samples1),
        reverse_work=energy1.energy(samples2) - (1.0 + energy2.energy(samples2)),
        implementation="pymbar"
    )
    assert free_energy1.item() == pytest.approx(free_energy2.item(), rel=1e-3)
    assert uncertainty1.item() == pytest.approx(uncertainty2.item(), rel=1e-3)
