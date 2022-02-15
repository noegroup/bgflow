"""Integration test for BoltzmannGenerator class"""


import pytest
import math
import torch
from bgflow.distribution.normal import NormalDistribution
from bgflow.distribution import TorchDistribution
from bgflow.nn.flow.base import Flow
from bgflow.bg import (
    BoltzmannGenerator, unnormalized_kl_div, unormalized_nll,
    log_weights, log_weights_given_latent, sampling_efficiency, effective_sample_size
)


@pytest.mark.parametrize("is_flow_exact", [True, False])
def test_bg_metrics(ctx, is_flow_exact):
    """Test Boltzmann generator metrics by transforming
    one normal distribution into another by a linear transformation.
    """
    torch.manual_seed(12345)

    # prior = standard normal
    prior = TorchDistribution(
        torch.distributions.Independent(
            torch.distributions.Normal(torch.tensor((0.,), **ctx), torch.tensor((1.,), **ctx)),
            1
        )
    )
    # target = (scale * standard normal + shift)
    target = TorchDistribution(
        torch.distributions.Independent(
            torch.distributions.Normal(torch.tensor((1.,), **ctx), torch.tensor((2.,), **ctx)),
            1
        )
    )

    class Shift(Flow):
        def __init__(self, shift=1.0, scale=2.0):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(shift))
            self.logscale = torch.nn.Parameter(torch.tensor(math.log(scale)))

        def _forward(self, x, **kwargs):
            return self.logscale.exp()*x + self.shift, self.logscale*torch.ones_like(x).sum(dim=-1, keepdim=True)

        def _inverse(self, x, **kwargs):
            return (x - self.shift)/self.logscale.exp(), -self.logscale*torch.ones_like(x).sum(dim=-1, keepdim=True)

    shift = 1. if is_flow_exact else 5.0
    scale = 2. if is_flow_exact else 0.01
    flow = Shift(shift, scale).to(**ctx)
    gen = BoltzmannGenerator(prior, flow, target)

    with torch.no_grad():
        z = prior.sample(10)
        x, dlogp = flow.forward(z)

        # energies
        assert is_flow_exact == torch.allclose(
            prior.energy(z) + dlogp,
            target.energy(x)
        )
        assert is_flow_exact == torch.allclose(
            gen.energy(x),
            target.energy(x)
        )

        # weights
        logw = log_weights(x, prior=prior, flow=flow, target=target)
        assert is_flow_exact == torch.allclose(
            logw, torch.log(1/len(logw)*torch.ones_like(logw)),
            atol=1e-4
        )
        assert torch.allclose(
            logw,
            log_weights_given_latent(x, z, dlogp, prior, target),
            atol=1e-4
        )
        assert torch.allclose(
            logw,
            gen.log_weights(x),
            atol=1e-4,
        )
        assert torch.allclose(
            logw,
            gen.log_weights_given_latent(x, z, dlogp),
            atol=1e-4
        )
        effsize = effective_sample_size(logw)
        assert is_flow_exact == torch.allclose(
            effsize, torch.tensor(10., **ctx), atol=1e-4
        )
        seff = sampling_efficiency(logw)
        assert is_flow_exact == torch.allclose(
            seff, torch.ones_like(seff), atol=1e-4
        )

    ## nll (gradient)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    opt.zero_grad()
    x = target.sample(1000)
    nll = unormalized_nll(prior, flow, x)
    assert torch.allclose(
        nll,
        gen.energy(x)
    )
    nll.mean().backward()
    # -- optimal gradients
    assert is_flow_exact == torch.allclose(flow.shift.grad, torch.zeros_like(flow.shift.grad), atol=1e-1)
    assert is_flow_exact == torch.allclose(flow.logscale.grad, torch.zeros_like(flow.logscale.grad), atol=1e-1)

    ## kld gradient
    opt.zero_grad()
    kld = unnormalized_kl_div(prior, flow, target, 1000)
    kld.mean().backward()
    # -- optimal gradients
    assert is_flow_exact == torch.allclose(flow.shift.grad, torch.zeros_like(flow.shift.grad), atol=1e-1)
    assert is_flow_exact == torch.allclose(flow.logscale.grad, torch.zeros_like(flow.logscale.grad), atol=1e-1)

    ## generator.kld
    opt.zero_grad()
    kld = gen.kldiv(1000)
    kld.mean().backward()
    # -- optimal gradients
    assert is_flow_exact == torch.allclose(flow.shift.grad, torch.zeros_like(flow.shift.grad), atol=1e-1)
    assert is_flow_exact == torch.allclose(flow.logscale.grad, torch.zeros_like(flow.logscale.grad), atol=1e-1)


def test_bg_basic(device, dtype):
    dim = 4
    mean = torch.zeros(dim, dtype=dtype, device=device)
    import bgflow as bg
    prior = bg.NormalDistribution(4, mean)
    # RealNVP
    flow = bg.SequentialFlow([
        bg.SplitFlow(dim//2),
        bg.CouplingFlow(
            bg.AffineTransformer(
                bg.DenseNet([dim//2, dim, dim//2]),
                bg.DenseNet([dim//2, dim, dim//2])
            )
        ),
        bg.SwapFlow(),
        bg.CouplingFlow(
            bg.AffineTransformer(
                bg.DenseNet([dim//2, dim, dim//2]),
                bg.DenseNet([dim//2, dim, dim//2])
            )
        ),
        bg.SwapFlow(),
        bg.MergeFlow(dim//2)
    ]).to(mean)
    target = bg.NormalDistribution(dim, mean)

    generator = bg.BoltzmannGenerator(
        prior, flow, target
    )

    # set parameters to 0 -> flow = id
    for p in generator.parameters():
        p.data.zero_()
    z = prior.sample(10)
    x, dlogp = flow.forward(z)
    assert torch.allclose(z,x)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp))

    # Test losses
    generator.zero_grad()
    kll = generator.kldiv(100000)
    kll.mean().backward()
    # gradients should be small, as the network is already optimal
    for p in generator.parameters():
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), rtol=0.0, atol=5e-2)

    generator.zero_grad()
    samples = target.sample(100000)
    nll = generator.energy(samples)
    nll.mean().backward()
    # gradients should be small, as the network is already optimal
    for p in generator.parameters():
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), rtol=0.0, atol=5e-2)

    # just testing the API for the following:
    generator.log_weights(samples)
    z, dlogp = flow.forward(samples, inverse=True)
    generator.log_weights_given_latent(samples, z, dlogp)
    generator.sample(10000)
    generator.force(z)

    # test trainers
    trainer = bg.KLTrainer(generator)
    trainer.train(100, samples)


def test_bg_basic_multiple(device, dtype):
    dim = 4
    mean = torch.zeros(dim//2, dtype=dtype, device=device)
    import bgflow as bg
    prior = bg.ProductDistribution([
        bg.NormalDistribution(dim//2, mean),
        bg.NormalDistribution(dim//2, mean)
    ])
    # RealNVP
    flow = bg.SequentialFlow([
        bg.CouplingFlow(
            bg.AffineTransformer(
                bg.DenseNet([dim//2, dim, dim//2]),
                bg.DenseNet([dim//2, dim, dim//2])
            )
        ),
        bg.SwapFlow(),
        bg.CouplingFlow(
            bg.AffineTransformer(
                bg.DenseNet([dim//2, dim, dim//2]),
                bg.DenseNet([dim//2, dim, dim//2])
            )
        ),
        bg.SwapFlow(),
    ]).to(mean)
    target = bg.ProductDistribution([
        bg.NormalDistribution(dim//2, mean),
        bg.NormalDistribution(dim//2, mean)
    ])

    generator = bg.BoltzmannGenerator(
        prior, flow, target
    )

    # set parameters to 0 -> flow = id
    for p in generator.parameters():
        p.data.zero_()
    z = prior.sample(10)
    *x, dlogp = flow.forward(*z)
    for zi, xi in zip(z,x):
        assert torch.allclose(zi, xi)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp))

    # Test losses
    generator.zero_grad()
    kll = generator.kldiv(100000)
    kll.mean().backward()
    # gradients should be small, as the network is already optimal
    for p in generator.parameters():
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), rtol=0.0, atol=5e-2)

    generator.zero_grad()
    samples = target.sample(100000)
    nll = generator.energy(*samples)
    nll.mean().backward()
    # gradients should be small, as the network is already optimal
    for p in generator.parameters():
        assert torch.allclose(p.grad, torch.zeros_like(p.grad), rtol=0.0, atol=5e-2)

    # just testing the API for the following:
    generator.log_weights(*samples)
    *z, dlogp = flow.forward(*samples, inverse=True)
    generator.log_weights_given_latent(samples, z, dlogp)
    generator.sample(10000)
    generator.force(*z)

    # test trainers
    trainer = bg.KLTrainer(generator)
    sampler = bg.ProductSampler([
        bg.DataSetSampler(samples[0]),
        bg.DataSetSampler(samples[1])
    ]).to(device=device, dtype=dtype)
    trainer.train(100, sampler)
