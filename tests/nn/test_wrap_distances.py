import torch
import numpy as np
from bgflow.nn.periodic import WrapDistances


def test_wrap_distances():
    positions = torch.tensor([[[0.,0.,0.],[0.,0.,1.],[0.,2.,0.]]])
    positions_flat = positions.view(positions.shape[0],-1)
    module = torch.nn.ReLU()
    wrapper = WrapDistances(module)
    result = wrapper.forward(positions_flat)
    expected = torch.tensor([[1.,2.,np.sqrt(5)]]).to(positions)
    assert torch.allclose(result, expected)
