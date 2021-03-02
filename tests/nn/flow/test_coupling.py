
import torch
from bgtorch import SplitFlow


def test_split_flow(device, dtype):
    tensor = torch.arange(0, 12., device=device, dtype=dtype).reshape(3,4)
    # pass or infer last size
    for sizes in ((2,), (2,2)):
        split = SplitFlow(*sizes, dim=-1)
        *result, dlogp = split.forward(tensor)
        assert torch.allclose(result[0], tensor[...,:2])
        assert torch.allclose(result[1], tensor[...,2:])
        assert dlogp.shape == (3,1)
    # dim != -1
    split = SplitFlow(2, dim=0)
    *result, dlogp = split.forward(tensor)
    assert torch.allclose(result[0], tensor[:2,...])
    assert torch.allclose(result[1], tensor[2:,...])
    assert dlogp.shape == (1,4)  #  <- this does not make sense yet until we allow event shapes
