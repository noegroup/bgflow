
# If this fails the example in the readme wont work!

def test_readme():
    import torch
    import matplotlib.pyplot as plt
    import bgflow as bg

    # define prior and target
    dim = 2
    prior = bg.NormalDistribution(dim)
    target = bg.DoubleWellEnergy(dim)

    # here we aggregate all layers of the flow
    layers = []
    layers.append(bg.SplitFlow(dim // 2))
    layers.append(bg.CouplingFlow(
        # we use a affine transformation to transform the RHS conditioned on the LHS
        bg.AffineTransformer(
            # use simple dense nets for the affine shift/scale
            shift_transformation=bg.DenseNet(
                [dim // 2, 4, dim // 2],
                activation=torch.nn.ReLU()
            ),
            scale_transformation=bg.DenseNet(
                [dim // 2, 4, dim // 2],
                activation=torch.nn.Tanh()
            )
        )
    ))
    layers.append(bg.InverseFlow(bg.SplitFlow(dim // 2)))

    # now define the flow as a sequence of all operations stored in layers
    flow = bg.SequentialFlow(layers)

    # The BG is defined by a prior, target and a flow
    generator = bg.BoltzmannGenerator(prior, flow, target)

    # sample from the BG
    samples = generator.sample(1000)
    plt.hist2d(
        samples[:, 0].detach().numpy(),
        samples[:, 1].detach().numpy(), bins=100
    )