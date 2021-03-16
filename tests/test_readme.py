import pytest
import torch
import matplotlib.pyplot as plt
from bgtorch import BoltzmannGenerator
from bgtorch.distribution import NormalDistribution
from bgtorch.distribution.energy import DoubleWellEnergy
from bgtorch.nn import DenseNet
from bgtorch.nn.flow import (SplitFlow, InverseFlow, SequentialFlow, CouplingFlow)
from bgtorch.nn.flow.transformer import AffineTransformer


# If this fails the example in the readme wont work!

def test_readme():
    # define prior and target
    dim = 2
    prior = NormalDistribution(dim)
    target = DoubleWellEnergy(dim)

    # here we aggregate all layers of the flow
    layers = []
    layers.append(SplitFlow(dim // 2))
    layers.append(CouplingFlow(
        # we use a affine transformation to transform the RHS conditioned on the LHS
        AffineTransformer(
            # use simple dense nets for the affine shift/scale
            shift_transformation=DenseNet([dim // 2, 4, dim // 2], activation=torch.nn.ReLU()),
            scale_transformation=DenseNet([dim // 2, 4, dim // 2], activation=torch.nn.Tanh())
        )
    ))
    layers.append(InverseFlow(SplitFlow(dim // 2)))

    # now define the flow as a sequence of all operations stored in layers
    flow = SequentialFlow(layers)

    # The BG is defined by a prior, target and a flow
    bg = BoltzmannGenerator(prior, flow, target)

    # sample from the BG
    samples = bg.sample(100000)
    plt.hist2d(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), bins=100);
