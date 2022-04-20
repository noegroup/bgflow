import torch
import numpy as np
import pytest
import bgflow

positions = torch.tensor([[0,0,0],[0,0,1],[0,2,0]])
module = torch.nn.ReLU()
wrapper = WrapDistances(module)
result = wrapper.forward(positions)
expected = torch.tensor([1,2,np.sqrt(3)])
assert torch.allclose(result, expected)