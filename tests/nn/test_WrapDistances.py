import torch
import numpy as np
import bgflow as bg
from bgflow.nn.periodic import WrapDistances
import pytest


positions = torch.tensor([[[0.,0.,0.],[0.,0.,1.],[0.,2.,0.]]])
positions_flat = positions.view(positions.shape[0],-1)
module = torch.nn.ReLU()
wrapper = WrapDistances(module, indices = [0, 1, 2, 3, 4, 5, 6, 7, 8])
#pytest.set_trace()
result = wrapper.forward(positions_flat)
expected = torch.tensor([[1.,2.,np.sqrt(5)]]).to(positions)
assert torch.allclose(result, expected)