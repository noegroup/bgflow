import pytest
import torch
from bgflow import UniformDistribution, NormalDistribution, TruncatedNormalDistribution, make_distribution


@pytest.mark.parametrize("prior_type", [UniformDistribution, NormalDistribution, TruncatedNormalDistribution])
def test_prior_factory(prior_type, ctx):
    prior = make_distribution(prior_type, 2, **ctx)
    samples = prior.sample(10)
    assert torch.device(samples.device) == torch.device(ctx["device"])
    assert samples.dtype == ctx["dtype"]
    assert samples.shape == (10, 2)


def test_prior_factory_with_kwargs(ctx):
    prior = make_distribution(UniformDistribution, 2, low=torch.tensor([2.0, 2.0]), high=torch.tensor([3.0, 3.0]), **ctx)
    samples = prior.sample(5)
    assert torch.device(samples.device) == torch.device(ctx["device"])
    assert samples.dtype == ctx["dtype"]
    assert (samples > 1.0).all()
