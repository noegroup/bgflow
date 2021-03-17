"""Integration test for BoltzmannGenerator class"""


import torch


def test_bg_basic(device, dtype):
    dim = 4
    mean = torch.zeros(dim, dtype=dtype, device=device)
    import bgtorch as bg
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
    import bgtorch as bg
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
    ])
    trainer.train(100, sampler)