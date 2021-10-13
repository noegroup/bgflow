
import torch
from bgflow.distribution import NormalDistribution


def test_free_energy_method(ctx):
    dim = 1
    energy1 = NormalDistribution(dim, mean=torch.zeros(dim, **ctx))
    energy2 = NormalDistribution(dim, mean=0.5*torch.ones(dim, **ctx))  # will be multiplied by e
    samples1 = energy1.sample(1000)
    samples2 = energy1.sample(2000)
    #forward_work =
