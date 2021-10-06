
import pytest
import torch
from bgflow import (
    DistributionTransferFlow, ConstrainGaussianFlow,
    CDFTransform, InverseFlow, TruncatedNormalDistribution
)
from torch.distributions import Normal


def test_distribution_transfer(ctx):
    src = Normal(torch.zeros(2, **ctx), torch.ones(2, **ctx))
    target = Normal(torch.ones(2, **ctx), torch.ones(2, **ctx))
    swap = DistributionTransferFlow(src, target)
    # forward
    out, dlogp = swap.forward(torch.zeros((2,2), **ctx))
    assert torch.allclose(out, torch.ones(2,2, **ctx))
    assert torch.allclose(dlogp, torch.zeros(2,1, **ctx))
    # inverse
    out2, dlogp = swap.forward(out, inverse=True)
    assert torch.allclose(out2, torch.zeros(2,2, **ctx))
    assert torch.allclose(dlogp, torch.zeros(2,1, **ctx))


def test_constrain_positivity(ctx):
    """Make sure that the bonds are obeyed."""
    torch.manual_seed(1)
    constrain_flow = ConstrainGaussianFlow(mu=torch.ones(10, **ctx), lower_bound=1e-10)
    samples = (1.0+torch.randn((10,10), **ctx)) * 1000.
    y, dlogp = constrain_flow.forward(samples)
    assert y.shape == (10, 10)
    assert dlogp.shape == (10, 1)
    assert (y >= 0.0).all()
    assert (dlogp.sum() < 0.0).all()


def test_constrain_slightly_pertubed(ctx):
    """Check that samples are not changed much when the bounds are generous."""
    torch.manual_seed(1)
    constrain_flow = ConstrainGaussianFlow(mu=torch.ones(10, **ctx), sigma=torch.ones(10, **ctx), lower_bound=-1000., upper_bound=1000.)
    samples = (1.0+torch.randn((10,10), **ctx))
    y, dlogp = constrain_flow.forward(samples)
    assert torch.allclose(samples, y, atol=1e-4, rtol=0.0)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp), atol=1e-4, rtol=0.0)

    x2, dlogp = constrain_flow.forward(y, inverse=True)
    assert torch.allclose(x2, y, atol=1e-4, rtol=0.0)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp), atol=1e-4, rtol=0.0)


def test_cdf_transform(ctx):
    input = torch.arange(0.1, 1.0, 0.1, **ctx)[:,None]
    input.requires_grad = True
    truncated_normal = TruncatedNormalDistribution(
        mu=torch.tensor([0.5], **ctx),
        upper_bound=torch.tensor([1.0], **ctx),
        is_learnable=True
    )
    flow = InverseFlow(CDFTransform((truncated_normal)))
    output, dlogp = flow.forward(input)
    assert output.mean().item() == pytest.approx(0.5)
    # try computing the grad
    output.mean().backward(create_graph=True)
    dlogp.mean().backward()


